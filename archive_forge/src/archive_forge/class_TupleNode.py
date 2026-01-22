from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
class TupleNode(SequenceNode):
    type = tuple_type
    is_partly_literal = False
    gil_message = 'Constructing Python tuple'

    def infer_type(self, env):
        if self.mult_factor or not self.args:
            return tuple_type
        arg_types = [arg.infer_type(env) for arg in self.args]
        if any((type.is_pyobject or type.is_memoryviewslice or type.is_unspecified or type.is_fused for type in arg_types)):
            return tuple_type
        return env.declare_tuple_type(self.pos, arg_types).type

    def analyse_types(self, env, skip_children=False):
        if self.is_literal:
            self.is_literal = False
        if self.is_partly_literal:
            self.is_partly_literal = False
        if len(self.args) == 0:
            self.is_temp = False
            self.is_literal = True
            return self
        if not skip_children:
            for i, arg in enumerate(self.args):
                if arg.is_starred:
                    arg.starred_expr_allowed_here = True
                self.args[i] = arg.analyse_types(env)
        if not self.mult_factor and (not any((arg.is_starred or arg.type.is_pyobject or arg.type.is_memoryviewslice or arg.type.is_fused for arg in self.args))):
            self.type = env.declare_tuple_type(self.pos, (arg.type for arg in self.args)).type
            self.is_temp = 1
            return self
        node = SequenceNode.analyse_types(self, env, skip_children=True)
        node = node._create_merge_node_if_necessary(env)
        if not node.is_sequence_constructor:
            return node
        if not all((child.is_literal for child in node.args)):
            return node
        if not node.mult_factor or (node.mult_factor.is_literal and isinstance(node.mult_factor.constant_result, _py_int_types)):
            node.is_temp = False
            node.is_literal = True
        else:
            if not node.mult_factor.type.is_pyobject and (not node.mult_factor.type.is_int):
                node.mult_factor = node.mult_factor.coerce_to_pyobject(env)
            node.is_temp = True
            node.is_partly_literal = True
        return node

    def analyse_as_type(self, env):
        if not self.args:
            return None
        item_types = [arg.analyse_as_type(env) for arg in self.args]
        if any((t is None for t in item_types)):
            return None
        entry = env.declare_tuple_type(self.pos, item_types)
        return entry.type

    def coerce_to(self, dst_type, env):
        if self.type.is_ctuple:
            if dst_type.is_ctuple and self.type.size == dst_type.size:
                return self.coerce_to_ctuple(dst_type, env)
            elif dst_type is tuple_type or dst_type is py_object_type:
                coerced_args = [arg.coerce_to_pyobject(env) for arg in self.args]
                return TupleNode(self.pos, args=coerced_args, type=tuple_type, mult_factor=self.mult_factor, is_temp=1).analyse_types(env, skip_children=True)
            else:
                return self.coerce_to_pyobject(env).coerce_to(dst_type, env)
        elif dst_type.is_ctuple and (not self.mult_factor):
            return self.coerce_to_ctuple(dst_type, env)
        else:
            return SequenceNode.coerce_to(self, dst_type, env)

    def as_list(self):
        t = ListNode(self.pos, args=self.args, mult_factor=self.mult_factor)
        if isinstance(self.constant_result, tuple):
            t.constant_result = list(self.constant_result)
        return t

    def is_simple(self):
        return True

    def nonlocally_immutable(self):
        return True

    def calculate_result_code(self):
        if len(self.args) > 0:
            return self.result_code
        else:
            return Naming.empty_tuple

    def calculate_constant_result(self):
        self.constant_result = tuple([arg.constant_result for arg in self.args])

    def compile_time_value(self, denv):
        values = self.compile_time_value_list(denv)
        try:
            return tuple(values)
        except Exception as e:
            self.compile_time_value_error(e)

    def generate_operation_code(self, code):
        if len(self.args) == 0:
            return
        if self.is_literal or self.is_partly_literal:
            dedup_key = make_dedup_key(self.type, [self.mult_factor if self.is_literal else None] + self.args)
            tuple_target = code.get_py_const(py_object_type, 'tuple', cleanup_level=2, dedup_key=dedup_key)
            const_code = code.get_cached_constants_writer(tuple_target)
            if const_code is not None:
                const_code.mark_pos(self.pos)
                self.generate_sequence_packing_code(const_code, tuple_target, plain=not self.is_literal)
                const_code.put_giveref(tuple_target, py_object_type)
            if self.is_literal:
                self.result_code = tuple_target
            elif self.mult_factor.type.is_int:
                code.globalstate.use_utility_code(UtilityCode.load_cached('PySequenceMultiply', 'ObjectHandling.c'))
                code.putln('%s = __Pyx_PySequence_Multiply(%s, %s); %s' % (self.result(), tuple_target, self.mult_factor.result(), code.error_goto_if_null(self.result(), self.pos)))
                self.generate_gotref(code)
            else:
                code.putln('%s = PyNumber_Multiply(%s, %s); %s' % (self.result(), tuple_target, self.mult_factor.py_result(), code.error_goto_if_null(self.result(), self.pos)))
                self.generate_gotref(code)
        else:
            self.type.entry.used = True
            self.generate_sequence_packing_code(code)
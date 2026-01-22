from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
class ParallelWithBlockNode(ParallelStatNode):
    """
    This node represents a 'with cython.parallel.parallel():' block
    """
    valid_keyword_arguments = ['num_threads']
    num_threads = None

    def analyse_declarations(self, env):
        super(ParallelWithBlockNode, self).analyse_declarations(env)
        if self.args:
            error(self.pos, 'cython.parallel.parallel() does not take positional arguments')

    def generate_execution_code(self, code):
        self.declare_closure_privates(code)
        self.setup_parallel_control_flow_block(code)
        code.putln('#ifdef _OPENMP')
        code.put('#pragma omp parallel ')
        if self.privates:
            privates = [e.cname for e in self.privates if not e.type.is_pyobject]
            code.put('private(%s)' % ', '.join(sorted(privates)))
        self.privatization_insertion_point = code.insertion_point()
        self.put_num_threads(code)
        code.putln('')
        code.putln('#endif /* _OPENMP */')
        code.begin_block()
        self.begin_parallel_block(code)
        self.initialize_privates_to_nan(code)
        code.funcstate.start_collecting_temps()
        self.body.generate_execution_code(code)
        self.trap_parallel_exit(code)
        self.privatize_temps(code)
        self.end_parallel_block(code)
        code.end_block()
        continue_ = code.label_used(code.continue_label)
        break_ = code.label_used(code.break_label)
        return_ = code.label_used(code.return_label)
        self.restore_labels(code)
        self.end_parallel_control_flow_block(code, break_=break_, continue_=continue_, return_=return_)
        self.release_closure_privates(code)
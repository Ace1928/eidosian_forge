from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
class IterationTransform(Visitor.EnvTransform):
    """Transform some common for-in loop patterns into efficient C loops:

    - for-in-dict loop becomes a while loop calling PyDict_Next()
    - for-in-enumerate is replaced by an external counter variable
    - for-in-range loop becomes a plain C for loop
    """

    def visit_PrimaryCmpNode(self, node):
        if node.is_ptr_contains():
            pos = node.pos
            result_ref = UtilNodes.ResultRefNode(node)
            if node.operand2.is_subscript:
                base_type = node.operand2.base.type.base_type
            else:
                base_type = node.operand2.type.base_type
            target_handle = UtilNodes.TempHandle(base_type)
            target = target_handle.ref(pos)
            cmp_node = ExprNodes.PrimaryCmpNode(pos, operator=u'==', operand1=node.operand1, operand2=target)
            if_body = Nodes.StatListNode(pos, stats=[Nodes.SingleAssignmentNode(pos, lhs=result_ref, rhs=ExprNodes.BoolNode(pos, value=1)), Nodes.BreakStatNode(pos)])
            if_node = Nodes.IfStatNode(pos, if_clauses=[Nodes.IfClauseNode(pos, condition=cmp_node, body=if_body)], else_clause=None)
            for_loop = UtilNodes.TempsBlockNode(pos, temps=[target_handle], body=Nodes.ForInStatNode(pos, target=target, iterator=ExprNodes.IteratorNode(node.operand2.pos, sequence=node.operand2), body=if_node, else_clause=Nodes.SingleAssignmentNode(pos, lhs=result_ref, rhs=ExprNodes.BoolNode(pos, value=0))))
            for_loop = for_loop.analyse_expressions(self.current_env())
            for_loop = self.visit(for_loop)
            new_node = UtilNodes.TempResultFromStatNode(result_ref, for_loop)
            if node.operator == 'not_in':
                new_node = ExprNodes.NotNode(pos, operand=new_node)
            return new_node
        else:
            self.visitchildren(node)
            return node

    def visit_ForInStatNode(self, node):
        self.visitchildren(node)
        return self._optimise_for_loop(node, node.iterator.sequence)

    def _optimise_for_loop(self, node, iterable, reversed=False):
        annotation_type = None
        if (iterable.is_name or iterable.is_attribute) and iterable.entry and iterable.entry.annotation:
            annotation = iterable.entry.annotation.expr
            if annotation.is_subscript:
                annotation = annotation.base
        if Builtin.dict_type in (iterable.type, annotation_type):
            if reversed:
                return node
            return self._transform_dict_iteration(node, dict_obj=iterable, method=None, keys=True, values=False)
        if Builtin.set_type in (iterable.type, annotation_type) or Builtin.frozenset_type in (iterable.type, annotation_type):
            if reversed:
                return node
            return self._transform_set_iteration(node, iterable)
        if iterable.type.is_ptr or iterable.type.is_array:
            return self._transform_carray_iteration(node, iterable, reversed=reversed)
        if iterable.type is Builtin.bytes_type:
            return self._transform_bytes_iteration(node, iterable, reversed=reversed)
        if iterable.type is Builtin.unicode_type:
            return self._transform_unicode_iteration(node, iterable, reversed=reversed)
        if iterable.type is Builtin.bytearray_type:
            return self._transform_indexable_iteration(node, iterable, is_mutable=True, reversed=reversed)
        if isinstance(iterable, ExprNodes.CoerceToPyTypeNode) and iterable.arg.type.is_memoryviewslice:
            return self._transform_indexable_iteration(node, iterable.arg, is_mutable=False, reversed=reversed)
        if not isinstance(iterable, ExprNodes.SimpleCallNode):
            return node
        if iterable.args is None:
            arg_count = iterable.arg_tuple and len(iterable.arg_tuple.args) or 0
        else:
            arg_count = len(iterable.args)
            if arg_count and iterable.self is not None:
                arg_count -= 1
        function = iterable.function
        if function.is_attribute and (not reversed) and (not arg_count):
            base_obj = iterable.self or function.obj
            method = function.attribute
            is_safe_iter = self.global_scope().context.language_level >= 3
            if not is_safe_iter and method in ('keys', 'values', 'items'):
                if isinstance(base_obj, ExprNodes.CallNode):
                    inner_function = base_obj.function
                    if inner_function.is_name and inner_function.name == 'dict' and inner_function.entry and inner_function.entry.is_builtin:
                        is_safe_iter = True
            keys = values = False
            if method == 'iterkeys' or (is_safe_iter and method == 'keys'):
                keys = True
            elif method == 'itervalues' or (is_safe_iter and method == 'values'):
                values = True
            elif method == 'iteritems' or (is_safe_iter and method == 'items'):
                keys = values = True
            if keys or values:
                return self._transform_dict_iteration(node, base_obj, method, keys, values)
        if iterable.self is None and function.is_name and function.entry and function.entry.is_builtin:
            if function.name == 'enumerate':
                if reversed:
                    return node
                return self._transform_enumerate_iteration(node, iterable)
            elif function.name == 'reversed':
                if reversed:
                    return node
                return self._transform_reversed_iteration(node, iterable)
        if Options.convert_range and 1 <= arg_count <= 3 and (iterable.self is None and function.is_name and (function.name in ('range', 'xrange')) and function.entry and function.entry.is_builtin):
            if node.target.type.is_int or node.target.type.is_enum:
                return self._transform_range_iteration(node, iterable, reversed=reversed)
            if node.target.type.is_pyobject:
                for arg in iterable.arg_tuple.args if iterable.args is None else iterable.args:
                    if isinstance(arg, ExprNodes.IntNode):
                        if arg.has_constant_result() and -2 ** 30 <= arg.constant_result < 2 ** 30:
                            continue
                    break
                else:
                    return self._transform_range_iteration(node, iterable, reversed=reversed)
        return node

    def _transform_reversed_iteration(self, node, reversed_function):
        args = reversed_function.arg_tuple.args
        if len(args) == 0:
            error(reversed_function.pos, 'reversed() requires an iterable argument')
            return node
        elif len(args) > 1:
            error(reversed_function.pos, 'reversed() takes exactly 1 argument')
            return node
        arg = args[0]
        if arg.type in (Builtin.tuple_type, Builtin.list_type):
            node.iterator.sequence = arg.as_none_safe_node("'NoneType' object is not iterable")
            node.iterator.reversed = True
            return node
        return self._optimise_for_loop(node, arg, reversed=True)

    def _transform_indexable_iteration(self, node, slice_node, is_mutable, reversed=False):
        """In principle can handle any iterable that Cython has a len() for and knows how to index"""
        unpack_temp_node = UtilNodes.LetRefNode(slice_node.as_none_safe_node("'NoneType' is not iterable"), may_hold_none=False, is_temp=True)
        start_node = ExprNodes.IntNode(node.pos, value='0', constant_result=0, type=PyrexTypes.c_py_ssize_t_type)

        def make_length_call():
            builtin_len = ExprNodes.NameNode(node.pos, name='len', entry=Builtin.builtin_scope.lookup('len'))
            return ExprNodes.SimpleCallNode(node.pos, function=builtin_len, args=[unpack_temp_node])
        length_temp = UtilNodes.LetRefNode(make_length_call(), type=PyrexTypes.c_py_ssize_t_type, is_temp=True)
        end_node = length_temp
        if reversed:
            relation1, relation2 = ('>', '>=')
            start_node, end_node = (end_node, start_node)
        else:
            relation1, relation2 = ('<=', '<')
        counter_ref = UtilNodes.LetRefNode(pos=node.pos, type=PyrexTypes.c_py_ssize_t_type)
        target_value = ExprNodes.IndexNode(slice_node.pos, base=unpack_temp_node, index=counter_ref)
        target_assign = Nodes.SingleAssignmentNode(pos=node.target.pos, lhs=node.target, rhs=target_value)
        env = self.current_env()
        new_directives = Options.copy_inherited_directives(env.directives, boundscheck=False, wraparound=False)
        target_assign = Nodes.CompilerDirectivesNode(target_assign.pos, directives=new_directives, body=target_assign)
        body = Nodes.StatListNode(node.pos, stats=[target_assign])
        if is_mutable:
            loop_length_reassign = Nodes.SingleAssignmentNode(node.pos, lhs=length_temp, rhs=make_length_call())
            body.stats.append(loop_length_reassign)
        loop_node = Nodes.ForFromStatNode(node.pos, bound1=start_node, relation1=relation1, target=counter_ref, relation2=relation2, bound2=end_node, step=None, body=body, else_clause=node.else_clause, from_range=True)
        ret = UtilNodes.LetNode(unpack_temp_node, UtilNodes.LetNode(length_temp, Nodes.ExprStatNode(node.pos, expr=UtilNodes.TempResultFromStatNode(counter_ref, loop_node)))).analyse_expressions(env)
        body.stats.insert(1, node.body)
        return ret
    PyBytes_AS_STRING_func_type = PyrexTypes.CFuncType(PyrexTypes.c_char_ptr_type, [PyrexTypes.CFuncTypeArg('s', Builtin.bytes_type, None)])
    PyBytes_GET_SIZE_func_type = PyrexTypes.CFuncType(PyrexTypes.c_py_ssize_t_type, [PyrexTypes.CFuncTypeArg('s', Builtin.bytes_type, None)])

    def _transform_bytes_iteration(self, node, slice_node, reversed=False):
        target_type = node.target.type
        if not target_type.is_int and target_type is not Builtin.bytes_type:
            return node
        unpack_temp_node = UtilNodes.LetRefNode(slice_node.as_none_safe_node("'NoneType' is not iterable"))
        slice_base_node = ExprNodes.PythonCapiCallNode(slice_node.pos, 'PyBytes_AS_STRING', self.PyBytes_AS_STRING_func_type, args=[unpack_temp_node], is_temp=0)
        len_node = ExprNodes.PythonCapiCallNode(slice_node.pos, 'PyBytes_GET_SIZE', self.PyBytes_GET_SIZE_func_type, args=[unpack_temp_node], is_temp=0)
        return UtilNodes.LetNode(unpack_temp_node, self._transform_carray_iteration(node, ExprNodes.SliceIndexNode(slice_node.pos, base=slice_base_node, start=None, step=None, stop=len_node, type=slice_base_node.type, is_temp=1), reversed=reversed))
    PyUnicode_READ_func_type = PyrexTypes.CFuncType(PyrexTypes.c_py_ucs4_type, [PyrexTypes.CFuncTypeArg('kind', PyrexTypes.c_int_type, None), PyrexTypes.CFuncTypeArg('data', PyrexTypes.c_void_ptr_type, None), PyrexTypes.CFuncTypeArg('index', PyrexTypes.c_py_ssize_t_type, None)])
    init_unicode_iteration_func_type = PyrexTypes.CFuncType(PyrexTypes.c_int_type, [PyrexTypes.CFuncTypeArg('s', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('length', PyrexTypes.c_py_ssize_t_ptr_type, None), PyrexTypes.CFuncTypeArg('data', PyrexTypes.c_void_ptr_ptr_type, None), PyrexTypes.CFuncTypeArg('kind', PyrexTypes.c_int_ptr_type, None)], exception_value='-1')

    def _transform_unicode_iteration(self, node, slice_node, reversed=False):
        if slice_node.is_literal:
            try:
                bytes_value = bytes_literal(slice_node.value.encode('latin1'), 'iso8859-1')
            except UnicodeEncodeError:
                pass
            else:
                bytes_slice = ExprNodes.SliceIndexNode(slice_node.pos, base=ExprNodes.BytesNode(slice_node.pos, value=bytes_value, constant_result=bytes_value, type=PyrexTypes.c_const_char_ptr_type).coerce_to(PyrexTypes.c_const_uchar_ptr_type, self.current_env()), start=None, stop=ExprNodes.IntNode(slice_node.pos, value=str(len(bytes_value)), constant_result=len(bytes_value), type=PyrexTypes.c_py_ssize_t_type), type=Builtin.unicode_type)
                return self._transform_carray_iteration(node, bytes_slice, reversed)
        unpack_temp_node = UtilNodes.LetRefNode(slice_node.as_none_safe_node("'NoneType' is not iterable"))
        start_node = ExprNodes.IntNode(node.pos, value='0', constant_result=0, type=PyrexTypes.c_py_ssize_t_type)
        length_temp = UtilNodes.TempHandle(PyrexTypes.c_py_ssize_t_type)
        end_node = length_temp.ref(node.pos)
        if reversed:
            relation1, relation2 = ('>', '>=')
            start_node, end_node = (end_node, start_node)
        else:
            relation1, relation2 = ('<=', '<')
        kind_temp = UtilNodes.TempHandle(PyrexTypes.c_int_type)
        data_temp = UtilNodes.TempHandle(PyrexTypes.c_void_ptr_type)
        counter_temp = UtilNodes.TempHandle(PyrexTypes.c_py_ssize_t_type)
        target_value = ExprNodes.PythonCapiCallNode(slice_node.pos, '__Pyx_PyUnicode_READ', self.PyUnicode_READ_func_type, args=[kind_temp.ref(slice_node.pos), data_temp.ref(slice_node.pos), counter_temp.ref(node.target.pos)], is_temp=False)
        if target_value.type != node.target.type:
            target_value = target_value.coerce_to(node.target.type, self.current_env())
        target_assign = Nodes.SingleAssignmentNode(pos=node.target.pos, lhs=node.target, rhs=target_value)
        body = Nodes.StatListNode(node.pos, stats=[target_assign, node.body])
        loop_node = Nodes.ForFromStatNode(node.pos, bound1=start_node, relation1=relation1, target=counter_temp.ref(node.target.pos), relation2=relation2, bound2=end_node, step=None, body=body, else_clause=node.else_clause, from_range=True)
        setup_node = Nodes.ExprStatNode(node.pos, expr=ExprNodes.PythonCapiCallNode(slice_node.pos, '__Pyx_init_unicode_iteration', self.init_unicode_iteration_func_type, args=[unpack_temp_node, ExprNodes.AmpersandNode(slice_node.pos, operand=length_temp.ref(slice_node.pos), type=PyrexTypes.c_py_ssize_t_ptr_type), ExprNodes.AmpersandNode(slice_node.pos, operand=data_temp.ref(slice_node.pos), type=PyrexTypes.c_void_ptr_ptr_type), ExprNodes.AmpersandNode(slice_node.pos, operand=kind_temp.ref(slice_node.pos), type=PyrexTypes.c_int_ptr_type)], is_temp=True, result_is_used=False, utility_code=UtilityCode.load_cached('unicode_iter', 'Optimize.c')))
        return UtilNodes.LetNode(unpack_temp_node, UtilNodes.TempsBlockNode(node.pos, temps=[counter_temp, length_temp, data_temp, kind_temp], body=Nodes.StatListNode(node.pos, stats=[setup_node, loop_node])))

    def _transform_carray_iteration(self, node, slice_node, reversed=False):
        neg_step = False
        if isinstance(slice_node, ExprNodes.SliceIndexNode):
            slice_base = slice_node.base
            start = filter_none_node(slice_node.start)
            stop = filter_none_node(slice_node.stop)
            step = None
            if not stop:
                if not slice_base.type.is_pyobject:
                    error(slice_node.pos, 'C array iteration requires known end index')
                return node
        elif slice_node.is_subscript:
            assert isinstance(slice_node.index, ExprNodes.SliceNode)
            slice_base = slice_node.base
            index = slice_node.index
            start = filter_none_node(index.start)
            stop = filter_none_node(index.stop)
            step = filter_none_node(index.step)
            if step:
                if not isinstance(step.constant_result, _py_int_types) or step.constant_result == 0 or (step.constant_result > 0 and (not stop)) or (step.constant_result < 0 and (not start)):
                    if not slice_base.type.is_pyobject:
                        error(step.pos, 'C array iteration requires known step size and end index')
                    return node
                else:
                    step_value = step.constant_result
                    if reversed:
                        step_value = -step_value
                    neg_step = step_value < 0
                    step = ExprNodes.IntNode(step.pos, type=PyrexTypes.c_py_ssize_t_type, value=str(abs(step_value)), constant_result=abs(step_value))
        elif slice_node.type.is_array:
            if slice_node.type.size is None:
                error(slice_node.pos, 'C array iteration requires known end index')
                return node
            slice_base = slice_node
            start = None
            stop = ExprNodes.IntNode(slice_node.pos, value=str(slice_node.type.size), type=PyrexTypes.c_py_ssize_t_type, constant_result=slice_node.type.size)
            step = None
        else:
            if not slice_node.type.is_pyobject:
                error(slice_node.pos, 'C array iteration requires known end index')
            return node
        if start:
            start = start.coerce_to(PyrexTypes.c_py_ssize_t_type, self.current_env())
        if stop:
            stop = stop.coerce_to(PyrexTypes.c_py_ssize_t_type, self.current_env())
        if stop is None:
            if neg_step:
                stop = ExprNodes.IntNode(slice_node.pos, value='-1', type=PyrexTypes.c_py_ssize_t_type, constant_result=-1)
            else:
                error(slice_node.pos, 'C array iteration requires known step size and end index')
                return node
        if reversed:
            if not start:
                start = ExprNodes.IntNode(slice_node.pos, value='0', constant_result=0, type=PyrexTypes.c_py_ssize_t_type)
            start, stop = (stop, start)
        ptr_type = slice_base.type
        if ptr_type.is_array:
            ptr_type = ptr_type.element_ptr_type()
        carray_ptr = slice_base.coerce_to_simple(self.current_env())
        if start and start.constant_result != 0:
            start_ptr_node = ExprNodes.AddNode(start.pos, operand1=carray_ptr, operator='+', operand2=start, type=ptr_type)
        else:
            start_ptr_node = carray_ptr
        if stop and stop.constant_result != 0:
            stop_ptr_node = ExprNodes.AddNode(stop.pos, operand1=ExprNodes.CloneNode(carray_ptr), operator='+', operand2=stop, type=ptr_type).coerce_to_simple(self.current_env())
        else:
            stop_ptr_node = ExprNodes.CloneNode(carray_ptr)
        counter = UtilNodes.TempHandle(ptr_type)
        counter_temp = counter.ref(node.target.pos)
        if slice_base.type.is_string and node.target.type.is_pyobject:
            if slice_node.type is Builtin.unicode_type:
                target_value = ExprNodes.CastNode(ExprNodes.DereferenceNode(node.target.pos, operand=counter_temp, type=ptr_type.base_type), PyrexTypes.c_py_ucs4_type).coerce_to(node.target.type, self.current_env())
            else:
                target_value = ExprNodes.SliceIndexNode(node.target.pos, start=ExprNodes.IntNode(node.target.pos, value='0', constant_result=0, type=PyrexTypes.c_int_type), stop=ExprNodes.IntNode(node.target.pos, value='1', constant_result=1, type=PyrexTypes.c_int_type), base=counter_temp, type=Builtin.bytes_type, is_temp=1)
        elif node.target.type.is_ptr and (not node.target.type.assignable_from(ptr_type.base_type)):
            target_value = counter_temp
        else:
            target_value = ExprNodes.IndexNode(node.target.pos, index=ExprNodes.IntNode(node.target.pos, value='0', constant_result=0, type=PyrexTypes.c_int_type), base=counter_temp, type=ptr_type.base_type)
        if target_value.type != node.target.type:
            target_value = target_value.coerce_to(node.target.type, self.current_env())
        target_assign = Nodes.SingleAssignmentNode(pos=node.target.pos, lhs=node.target, rhs=target_value)
        body = Nodes.StatListNode(node.pos, stats=[target_assign, node.body])
        relation1, relation2 = self._find_for_from_node_relations(neg_step, reversed)
        for_node = Nodes.ForFromStatNode(node.pos, bound1=start_ptr_node, relation1=relation1, target=counter_temp, relation2=relation2, bound2=stop_ptr_node, step=step, body=body, else_clause=node.else_clause, from_range=True)
        return UtilNodes.TempsBlockNode(node.pos, temps=[counter], body=for_node)

    def _transform_enumerate_iteration(self, node, enumerate_function):
        args = enumerate_function.arg_tuple.args
        if len(args) == 0:
            error(enumerate_function.pos, 'enumerate() requires an iterable argument')
            return node
        elif len(args) > 2:
            error(enumerate_function.pos, 'enumerate() takes at most 2 arguments')
            return node
        if not node.target.is_sequence_constructor:
            return node
        targets = node.target.args
        if len(targets) != 2:
            return node
        enumerate_target, iterable_target = targets
        counter_type = enumerate_target.type
        if not counter_type.is_pyobject and (not counter_type.is_int):
            return node
        if len(args) == 2:
            start = unwrap_coerced_node(args[1]).coerce_to(counter_type, self.current_env())
        else:
            start = ExprNodes.IntNode(enumerate_function.pos, value='0', type=counter_type, constant_result=0)
        temp = UtilNodes.LetRefNode(start)
        inc_expression = ExprNodes.AddNode(enumerate_function.pos, operand1=temp, operand2=ExprNodes.IntNode(node.pos, value='1', type=counter_type, constant_result=1), operator='+', type=counter_type, is_temp=counter_type.is_pyobject)
        loop_body = [Nodes.SingleAssignmentNode(pos=enumerate_target.pos, lhs=enumerate_target, rhs=temp), Nodes.SingleAssignmentNode(pos=enumerate_target.pos, lhs=temp, rhs=inc_expression)]
        if isinstance(node.body, Nodes.StatListNode):
            node.body.stats = loop_body + node.body.stats
        else:
            loop_body.append(node.body)
            node.body = Nodes.StatListNode(node.body.pos, stats=loop_body)
        node.target = iterable_target
        node.item = node.item.coerce_to(iterable_target.type, self.current_env())
        node.iterator.sequence = args[0]
        return UtilNodes.LetNode(temp, self._optimise_for_loop(node, node.iterator.sequence))

    def _find_for_from_node_relations(self, neg_step_value, reversed):
        if reversed:
            if neg_step_value:
                return ('<', '<=')
            else:
                return ('>', '>=')
        elif neg_step_value:
            return ('>=', '>')
        else:
            return ('<=', '<')

    def _transform_range_iteration(self, node, range_function, reversed=False):
        args = range_function.arg_tuple.args
        if len(args) < 3:
            step_pos = range_function.pos
            step_value = 1
            step = ExprNodes.IntNode(step_pos, value='1', constant_result=1)
        else:
            step = args[2]
            step_pos = step.pos
            if not isinstance(step.constant_result, _py_int_types):
                return node
            step_value = step.constant_result
            if step_value == 0:
                return node
            step = ExprNodes.IntNode(step_pos, value=str(step_value), constant_result=step_value)
        if len(args) == 1:
            bound1 = ExprNodes.IntNode(range_function.pos, value='0', constant_result=0)
            bound2 = args[0].coerce_to_integer(self.current_env())
        else:
            bound1 = args[0].coerce_to_integer(self.current_env())
            bound2 = args[1].coerce_to_integer(self.current_env())
        relation1, relation2 = self._find_for_from_node_relations(step_value < 0, reversed)
        bound2_ref_node = None
        if reversed:
            bound1, bound2 = (bound2, bound1)
            abs_step = abs(step_value)
            if abs_step != 1:
                if isinstance(bound1.constant_result, _py_int_types) and isinstance(bound2.constant_result, _py_int_types):
                    if step_value < 0:
                        begin_value = bound2.constant_result
                        end_value = bound1.constant_result
                        bound1_value = begin_value - abs_step * ((begin_value - end_value - 1) // abs_step) - 1
                    else:
                        begin_value = bound1.constant_result
                        end_value = bound2.constant_result
                        bound1_value = end_value + abs_step * ((begin_value - end_value - 1) // abs_step) + 1
                    bound1 = ExprNodes.IntNode(bound1.pos, value=str(bound1_value), constant_result=bound1_value, type=PyrexTypes.spanning_type(bound1.type, bound2.type))
                else:
                    bound2_ref_node = UtilNodes.LetRefNode(bound2)
                    bound1 = self._build_range_step_calculation(bound1, bound2_ref_node, step, step_value)
        if step_value < 0:
            step_value = -step_value
        step.value = str(step_value)
        step.constant_result = step_value
        step = step.coerce_to_integer(self.current_env())
        if not bound2.is_literal:
            bound2_is_temp = True
            bound2 = bound2_ref_node or UtilNodes.LetRefNode(bound2)
        else:
            bound2_is_temp = False
        for_node = Nodes.ForFromStatNode(node.pos, target=node.target, bound1=bound1, relation1=relation1, relation2=relation2, bound2=bound2, step=step, body=node.body, else_clause=node.else_clause, from_range=True)
        for_node.set_up_loop(self.current_env())
        if bound2_is_temp:
            for_node = UtilNodes.LetNode(bound2, for_node)
        return for_node

    def _build_range_step_calculation(self, bound1, bound2_ref_node, step, step_value):
        abs_step = abs(step_value)
        spanning_type = PyrexTypes.spanning_type(bound1.type, bound2_ref_node.type)
        if step.type.is_int and abs_step < 32767:
            spanning_step_type = PyrexTypes.spanning_type(spanning_type, PyrexTypes.c_int_type)
        else:
            spanning_step_type = PyrexTypes.spanning_type(spanning_type, step.type)
        if step_value < 0:
            begin_value = bound2_ref_node
            end_value = bound1
            final_op = '-'
        else:
            begin_value = bound1
            end_value = bound2_ref_node
            final_op = '+'
        step_calculation_node = ExprNodes.binop_node(bound1.pos, operand1=ExprNodes.binop_node(bound1.pos, operand1=bound2_ref_node, operator=final_op, operand2=ExprNodes.MulNode(bound1.pos, operand1=ExprNodes.IntNode(bound1.pos, value=str(abs_step), constant_result=abs_step, type=spanning_step_type), operator='*', operand2=ExprNodes.DivNode(bound1.pos, operand1=ExprNodes.SubNode(bound1.pos, operand1=ExprNodes.SubNode(bound1.pos, operand1=begin_value, operator='-', operand2=end_value, type=spanning_type), operator='-', operand2=ExprNodes.IntNode(bound1.pos, value='1', constant_result=1), type=spanning_step_type), operator='//', operand2=ExprNodes.IntNode(bound1.pos, value=str(abs_step), constant_result=abs_step, type=spanning_step_type), type=spanning_step_type), type=spanning_step_type), type=spanning_step_type), operator=final_op, operand2=ExprNodes.IntNode(bound1.pos, value='1', constant_result=1), type=spanning_type)
        return step_calculation_node

    def _transform_dict_iteration(self, node, dict_obj, method, keys, values):
        temps = []
        temp = UtilNodes.TempHandle(PyrexTypes.py_object_type)
        temps.append(temp)
        dict_temp = temp.ref(dict_obj.pos)
        temp = UtilNodes.TempHandle(PyrexTypes.c_py_ssize_t_type)
        temps.append(temp)
        pos_temp = temp.ref(node.pos)
        key_target = value_target = tuple_target = None
        if keys and values:
            if node.target.is_sequence_constructor:
                if len(node.target.args) == 2:
                    key_target, value_target = node.target.args
                else:
                    return node
            else:
                tuple_target = node.target
        elif keys:
            key_target = node.target
        else:
            value_target = node.target
        if isinstance(node.body, Nodes.StatListNode):
            body = node.body
        else:
            body = Nodes.StatListNode(pos=node.body.pos, stats=[node.body])
        dict_len_temp = UtilNodes.TempHandle(PyrexTypes.c_py_ssize_t_type)
        temps.append(dict_len_temp)
        dict_len_temp_addr = ExprNodes.AmpersandNode(node.pos, operand=dict_len_temp.ref(dict_obj.pos), type=PyrexTypes.c_ptr_type(dict_len_temp.type))
        temp = UtilNodes.TempHandle(PyrexTypes.c_int_type)
        temps.append(temp)
        is_dict_temp = temp.ref(node.pos)
        is_dict_temp_addr = ExprNodes.AmpersandNode(node.pos, operand=is_dict_temp, type=PyrexTypes.c_ptr_type(temp.type))
        iter_next_node = Nodes.DictIterationNextNode(dict_temp, dict_len_temp.ref(dict_obj.pos), pos_temp, key_target, value_target, tuple_target, is_dict_temp)
        iter_next_node = iter_next_node.analyse_expressions(self.current_env())
        body.stats[0:0] = [iter_next_node]
        if method:
            method_node = ExprNodes.StringNode(dict_obj.pos, is_identifier=True, value=method)
            dict_obj = dict_obj.as_none_safe_node("'NoneType' object has no attribute '%{0}s'".format('.30' if len(method) <= 30 else ''), error='PyExc_AttributeError', format_args=[method])
        else:
            method_node = ExprNodes.NullNode(dict_obj.pos)
            dict_obj = dict_obj.as_none_safe_node("'NoneType' object is not iterable")

        def flag_node(value):
            value = value and 1 or 0
            return ExprNodes.IntNode(node.pos, value=str(value), constant_result=value)
        result_code = [Nodes.SingleAssignmentNode(node.pos, lhs=pos_temp, rhs=ExprNodes.IntNode(node.pos, value='0', constant_result=0)), Nodes.SingleAssignmentNode(dict_obj.pos, lhs=dict_temp, rhs=ExprNodes.PythonCapiCallNode(dict_obj.pos, '__Pyx_dict_iterator', self.PyDict_Iterator_func_type, utility_code=UtilityCode.load_cached('dict_iter', 'Optimize.c'), args=[dict_obj, flag_node(dict_obj.type is Builtin.dict_type), method_node, dict_len_temp_addr, is_dict_temp_addr], is_temp=True)), Nodes.WhileStatNode(node.pos, condition=None, body=body, else_clause=node.else_clause)]
        return UtilNodes.TempsBlockNode(node.pos, temps=temps, body=Nodes.StatListNode(node.pos, stats=result_code))
    PyDict_Iterator_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('dict', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('is_dict', PyrexTypes.c_int_type, None), PyrexTypes.CFuncTypeArg('method_name', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('p_orig_length', PyrexTypes.c_py_ssize_t_ptr_type, None), PyrexTypes.CFuncTypeArg('p_is_dict', PyrexTypes.c_int_ptr_type, None)])
    PySet_Iterator_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('set', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('is_set', PyrexTypes.c_int_type, None), PyrexTypes.CFuncTypeArg('p_orig_length', PyrexTypes.c_py_ssize_t_ptr_type, None), PyrexTypes.CFuncTypeArg('p_is_set', PyrexTypes.c_int_ptr_type, None)])

    def _transform_set_iteration(self, node, set_obj):
        temps = []
        temp = UtilNodes.TempHandle(PyrexTypes.py_object_type)
        temps.append(temp)
        set_temp = temp.ref(set_obj.pos)
        temp = UtilNodes.TempHandle(PyrexTypes.c_py_ssize_t_type)
        temps.append(temp)
        pos_temp = temp.ref(node.pos)
        if isinstance(node.body, Nodes.StatListNode):
            body = node.body
        else:
            body = Nodes.StatListNode(pos=node.body.pos, stats=[node.body])
        set_len_temp = UtilNodes.TempHandle(PyrexTypes.c_py_ssize_t_type)
        temps.append(set_len_temp)
        set_len_temp_addr = ExprNodes.AmpersandNode(node.pos, operand=set_len_temp.ref(set_obj.pos), type=PyrexTypes.c_ptr_type(set_len_temp.type))
        temp = UtilNodes.TempHandle(PyrexTypes.c_int_type)
        temps.append(temp)
        is_set_temp = temp.ref(node.pos)
        is_set_temp_addr = ExprNodes.AmpersandNode(node.pos, operand=is_set_temp, type=PyrexTypes.c_ptr_type(temp.type))
        value_target = node.target
        iter_next_node = Nodes.SetIterationNextNode(set_temp, set_len_temp.ref(set_obj.pos), pos_temp, value_target, is_set_temp)
        iter_next_node = iter_next_node.analyse_expressions(self.current_env())
        body.stats[0:0] = [iter_next_node]

        def flag_node(value):
            value = value and 1 or 0
            return ExprNodes.IntNode(node.pos, value=str(value), constant_result=value)
        result_code = [Nodes.SingleAssignmentNode(node.pos, lhs=pos_temp, rhs=ExprNodes.IntNode(node.pos, value='0', constant_result=0)), Nodes.SingleAssignmentNode(set_obj.pos, lhs=set_temp, rhs=ExprNodes.PythonCapiCallNode(set_obj.pos, '__Pyx_set_iterator', self.PySet_Iterator_func_type, utility_code=UtilityCode.load_cached('set_iter', 'Optimize.c'), args=[set_obj, flag_node(set_obj.type is Builtin.set_type), set_len_temp_addr, is_set_temp_addr], is_temp=True)), Nodes.WhileStatNode(node.pos, condition=None, body=body, else_clause=node.else_clause)]
        return UtilNodes.TempsBlockNode(node.pos, temps=temps, body=Nodes.StatListNode(node.pos, stats=result_code))
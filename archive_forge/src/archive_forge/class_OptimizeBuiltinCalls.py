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
class OptimizeBuiltinCalls(Visitor.NodeRefCleanupMixin, Visitor.MethodDispatcherTransform):
    """Optimize some common methods calls and instantiation patterns
    for builtin types *after* the type analysis phase.

    Running after type analysis, this transform can only perform
    function replacements that do not alter the function return type
    in a way that was not anticipated by the type analysis.
    """

    def visit_PyTypeTestNode(self, node):
        """Flatten redundant type checks after tree changes.
        """
        self.visitchildren(node)
        return node.reanalyse()

    def _visit_TypecastNode(self, node):
        """
        Drop redundant type casts.
        """
        self.visitchildren(node)
        if node.type == node.operand.type:
            return node.operand
        return node

    def visit_ExprStatNode(self, node):
        """
        Drop dead code and useless coercions.
        """
        self.visitchildren(node)
        if isinstance(node.expr, ExprNodes.CoerceToPyTypeNode):
            node.expr = node.expr.arg
        expr = node.expr
        if expr is None or expr.is_none or expr.is_literal:
            return None
        if expr.is_name and expr.entry and (expr.entry.is_local or expr.entry.is_arg):
            return None
        return node

    def visit_CoerceToBooleanNode(self, node):
        """Drop redundant conversion nodes after tree changes.
        """
        self.visitchildren(node)
        arg = node.arg
        if isinstance(arg, ExprNodes.PyTypeTestNode):
            arg = arg.arg
        if isinstance(arg, ExprNodes.CoerceToPyTypeNode):
            if arg.type in (PyrexTypes.py_object_type, Builtin.bool_type):
                return arg.arg.coerce_to_boolean(self.current_env())
        return node
    PyNumber_Float_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('o', PyrexTypes.py_object_type, None)])

    def visit_CoerceToPyTypeNode(self, node):
        """Drop redundant conversion nodes after tree changes."""
        self.visitchildren(node)
        arg = node.arg
        if isinstance(arg, ExprNodes.CoerceFromPyTypeNode):
            arg = arg.arg
        if isinstance(arg, ExprNodes.PythonCapiCallNode):
            if arg.function.name == 'float' and len(arg.args) == 1:
                func_arg = arg.args[0]
                if func_arg.type is Builtin.float_type:
                    return func_arg.as_none_safe_node("float() argument must be a string or a number, not 'NoneType'")
                elif func_arg.type.is_pyobject and arg.function.cname == '__Pyx_PyObject_AsDouble':
                    return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PyNumber_Float', self.PyNumber_Float_func_type, args=[func_arg], py_name='float', is_temp=node.is_temp, utility_code=UtilityCode.load_cached('pynumber_float', 'TypeConversion.c'), result_is_used=node.result_is_used).coerce_to(node.type, self.current_env())
        return node

    def visit_CoerceFromPyTypeNode(self, node):
        """Drop redundant conversion nodes after tree changes.

        Also, optimise away calls to Python's builtin int() and
        float() if the result is going to be coerced back into a C
        type anyway.
        """
        self.visitchildren(node)
        arg = node.arg
        if not arg.type.is_pyobject:
            if node.type != arg.type:
                arg = arg.coerce_to(node.type, self.current_env())
            return arg
        if isinstance(arg, ExprNodes.PyTypeTestNode):
            arg = arg.arg
        if arg.is_literal:
            if node.type.is_int and isinstance(arg, ExprNodes.IntNode) or (node.type.is_float and isinstance(arg, ExprNodes.FloatNode)) or (node.type.is_int and isinstance(arg, ExprNodes.BoolNode)):
                return arg.coerce_to(node.type, self.current_env())
        elif isinstance(arg, ExprNodes.CoerceToPyTypeNode):
            if arg.type is PyrexTypes.py_object_type:
                if node.type.assignable_from(arg.arg.type):
                    return arg.arg.coerce_to(node.type, self.current_env())
            elif arg.type is Builtin.unicode_type:
                if arg.arg.type.is_unicode_char and node.type.is_unicode_char:
                    return arg.arg.coerce_to(node.type, self.current_env())
        elif isinstance(arg, ExprNodes.SimpleCallNode):
            if node.type.is_int or node.type.is_float:
                return self._optimise_numeric_cast_call(node, arg)
        elif arg.is_subscript:
            index_node = arg.index
            if isinstance(index_node, ExprNodes.CoerceToPyTypeNode):
                index_node = index_node.arg
            if index_node.type.is_int:
                return self._optimise_int_indexing(node, arg, index_node)
        return node
    PyBytes_GetItemInt_func_type = PyrexTypes.CFuncType(PyrexTypes.c_char_type, [PyrexTypes.CFuncTypeArg('bytes', Builtin.bytes_type, None), PyrexTypes.CFuncTypeArg('index', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('check_bounds', PyrexTypes.c_int_type, None)], exception_value='((char)-1)', exception_check=True)

    def _optimise_int_indexing(self, coerce_node, arg, index_node):
        env = self.current_env()
        bound_check_bool = env.directives['boundscheck'] and 1 or 0
        if arg.base.type is Builtin.bytes_type:
            if coerce_node.type in (PyrexTypes.c_char_type, PyrexTypes.c_uchar_type):
                bound_check_node = ExprNodes.IntNode(coerce_node.pos, value=str(bound_check_bool), constant_result=bound_check_bool)
                node = ExprNodes.PythonCapiCallNode(coerce_node.pos, '__Pyx_PyBytes_GetItemInt', self.PyBytes_GetItemInt_func_type, args=[arg.base.as_none_safe_node("'NoneType' object is not subscriptable"), index_node.coerce_to(PyrexTypes.c_py_ssize_t_type, env), bound_check_node], is_temp=True, utility_code=UtilityCode.load_cached('bytes_index', 'StringTools.c'))
                if coerce_node.type is not PyrexTypes.c_char_type:
                    node = node.coerce_to(coerce_node.type, env)
                return node
        return coerce_node
    float_float_func_types = dict(((float_type, PyrexTypes.CFuncType(float_type, [PyrexTypes.CFuncTypeArg('arg', float_type, None)])) for float_type in (PyrexTypes.c_float_type, PyrexTypes.c_double_type, PyrexTypes.c_longdouble_type)))

    def _optimise_numeric_cast_call(self, node, arg):
        function = arg.function
        args = None
        if isinstance(arg, ExprNodes.PythonCapiCallNode):
            args = arg.args
        elif isinstance(function, ExprNodes.NameNode):
            if function.type.is_builtin_type and isinstance(arg.arg_tuple, ExprNodes.TupleNode):
                args = arg.arg_tuple.args
        if args is None or len(args) != 1:
            return node
        func_arg = args[0]
        if isinstance(func_arg, ExprNodes.CoerceToPyTypeNode):
            func_arg = func_arg.arg
        elif func_arg.type.is_pyobject:
            return node
        if function.name == 'int':
            if func_arg.type.is_int or node.type.is_int:
                if func_arg.type == node.type:
                    return func_arg
                elif func_arg.type in (PyrexTypes.c_py_ucs4_type, PyrexTypes.c_py_unicode_type):
                    return self._pyucs4_to_number(node, function.name, func_arg)
                elif node.type.assignable_from(func_arg.type) or func_arg.type.is_float:
                    return ExprNodes.TypecastNode(node.pos, operand=func_arg, type=node.type)
            elif func_arg.type.is_float and node.type.is_numeric:
                if func_arg.type.math_h_modifier == 'l':
                    truncl = '__Pyx_truncl'
                else:
                    truncl = 'trunc' + func_arg.type.math_h_modifier
                return ExprNodes.PythonCapiCallNode(node.pos, truncl, func_type=self.float_float_func_types[func_arg.type], args=[func_arg], py_name='int', is_temp=node.is_temp, result_is_used=node.result_is_used).coerce_to(node.type, self.current_env())
        elif function.name == 'float':
            if func_arg.type.is_float or node.type.is_float:
                if func_arg.type == node.type:
                    return func_arg
                elif func_arg.type in (PyrexTypes.c_py_ucs4_type, PyrexTypes.c_py_unicode_type):
                    return self._pyucs4_to_number(node, function.name, func_arg)
                elif node.type.assignable_from(func_arg.type) or func_arg.type.is_float:
                    return ExprNodes.TypecastNode(node.pos, operand=func_arg, type=node.type)
        return node
    pyucs4_int_func_type = PyrexTypes.CFuncType(PyrexTypes.c_int_type, [PyrexTypes.CFuncTypeArg('arg', PyrexTypes.c_py_ucs4_type, None)], exception_value='-1')
    pyucs4_double_func_type = PyrexTypes.CFuncType(PyrexTypes.c_double_type, [PyrexTypes.CFuncTypeArg('arg', PyrexTypes.c_py_ucs4_type, None)], exception_value='-1.0')

    def _pyucs4_to_number(self, node, py_type_name, func_arg):
        assert py_type_name in ('int', 'float')
        return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_int_from_UCS4' if py_type_name == 'int' else '__Pyx_double_from_UCS4', func_type=self.pyucs4_int_func_type if py_type_name == 'int' else self.pyucs4_double_func_type, args=[func_arg], py_name=py_type_name, is_temp=node.is_temp, result_is_used=node.result_is_used, utility_code=UtilityCode.load_cached('int_pyucs4' if py_type_name == 'int' else 'float_pyucs4', 'Builtins.c')).coerce_to(node.type, self.current_env())

    def _error_wrong_arg_count(self, function_name, node, args, expected=None):
        if not expected:
            arg_str = ''
        elif isinstance(expected, basestring) or expected > 1:
            arg_str = '...'
        elif expected == 1:
            arg_str = 'x'
        else:
            arg_str = ''
        if expected is not None:
            expected_str = 'expected %s, ' % expected
        else:
            expected_str = ''
        error(node.pos, '%s(%s) called with wrong number of args, %sfound %d' % (function_name, arg_str, expected_str, len(args)))

    def _handle_function(self, node, function_name, function, arg_list, kwargs):
        return node

    def _handle_method(self, node, type_name, attr_name, function, arg_list, is_unbound_method, kwargs):
        """
        Try to inject C-API calls for unbound method calls to builtin types.
        While the method declarations in Builtin.py already handle this, we
        can additionally resolve bound and unbound methods here that were
        assigned to variables ahead of time.
        """
        if kwargs:
            return node
        if not function or not function.is_attribute or (not function.obj.is_name):
            return node
        type_entry = self.current_env().lookup(type_name)
        if not type_entry:
            return node
        method = ExprNodes.AttributeNode(node.function.pos, obj=ExprNodes.NameNode(function.pos, name=type_name, entry=type_entry, type=type_entry.type), attribute=attr_name, is_called=True).analyse_as_type_attribute(self.current_env())
        if method is None:
            return self._optimise_generic_builtin_method_call(node, attr_name, function, arg_list, is_unbound_method)
        args = node.args
        if args is None and node.arg_tuple:
            args = node.arg_tuple.args
        call_node = ExprNodes.SimpleCallNode(node.pos, function=method, args=args)
        if not is_unbound_method:
            call_node.self = function.obj
        call_node.analyse_c_function_call(self.current_env())
        call_node.analysed = True
        return call_node.coerce_to(node.type, self.current_env())

    def _optimise_generic_builtin_method_call(self, node, attr_name, function, arg_list, is_unbound_method):
        """
        Try to inject an unbound method call for a call to a method of a known builtin type.
        This enables caching the underlying C function of the method at runtime.
        """
        arg_count = len(arg_list)
        if is_unbound_method or arg_count >= 3 or (not (function.is_attribute and function.is_py_attr)):
            return node
        if not function.obj.type.is_builtin_type:
            return node
        if function.obj.type.name in ('basestring', 'type'):
            return node
        return ExprNodes.CachedBuiltinMethodCallNode(node, function.obj, attr_name, arg_list)
    PyObject_String_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('obj', PyrexTypes.py_object_type, None)])

    def _handle_simple_function_str(self, node, function, pos_args):
        """Optimize single argument calls to str().
        """
        if len(pos_args) != 1:
            if len(pos_args) == 0:
                return ExprNodes.StringNode(node.pos, value=EncodedString(), constant_result='')
            return node
        arg = pos_args[0]
        if arg.type is Builtin.str_type:
            if not arg.may_be_none():
                return arg
            cname = '__Pyx_PyStr_Str'
            utility_code = UtilityCode.load_cached('PyStr_Str', 'StringTools.c')
        else:
            cname = '__Pyx_PyObject_Str'
            utility_code = UtilityCode.load_cached('PyObject_Str', 'StringTools.c')
        return ExprNodes.PythonCapiCallNode(node.pos, cname, self.PyObject_String_func_type, args=pos_args, is_temp=node.is_temp, utility_code=utility_code, py_name='str')
    PyObject_Unicode_func_type = PyrexTypes.CFuncType(Builtin.unicode_type, [PyrexTypes.CFuncTypeArg('obj', PyrexTypes.py_object_type, None)])

    def _handle_simple_function_unicode(self, node, function, pos_args):
        """Optimise single argument calls to unicode().
        """
        if len(pos_args) != 1:
            if len(pos_args) == 0:
                return ExprNodes.UnicodeNode(node.pos, value=EncodedString(), constant_result=u'')
            return node
        arg = pos_args[0]
        if arg.type is Builtin.unicode_type:
            if not arg.may_be_none():
                return arg
            cname = '__Pyx_PyUnicode_Unicode'
            utility_code = UtilityCode.load_cached('PyUnicode_Unicode', 'StringTools.c')
        else:
            cname = '__Pyx_PyObject_Unicode'
            utility_code = UtilityCode.load_cached('PyObject_Unicode', 'StringTools.c')
        return ExprNodes.PythonCapiCallNode(node.pos, cname, self.PyObject_Unicode_func_type, args=pos_args, is_temp=node.is_temp, utility_code=utility_code, py_name='unicode')

    def visit_FormattedValueNode(self, node):
        """Simplify or avoid plain string formatting of a unicode value.
        This seems misplaced here, but plain unicode formatting is essentially
        a call to the unicode() builtin, which is optimised right above.
        """
        self.visitchildren(node)
        if node.value.type is Builtin.unicode_type and (not node.c_format_spec) and (not node.format_spec):
            if not node.conversion_char or node.conversion_char == 's':
                return self._handle_simple_function_unicode(node, None, [node.value])
        return node
    PyDict_Copy_func_type = PyrexTypes.CFuncType(Builtin.dict_type, [PyrexTypes.CFuncTypeArg('dict', Builtin.dict_type, None)])

    def _handle_simple_function_dict(self, node, function, pos_args):
        """Replace dict(some_dict) by PyDict_Copy(some_dict).
        """
        if len(pos_args) != 1:
            return node
        arg = pos_args[0]
        if arg.type is Builtin.dict_type:
            arg = arg.as_none_safe_node("'NoneType' is not iterable")
            return ExprNodes.PythonCapiCallNode(node.pos, 'PyDict_Copy', self.PyDict_Copy_func_type, args=[arg], is_temp=node.is_temp)
        return node
    PySequence_List_func_type = PyrexTypes.CFuncType(Builtin.list_type, [PyrexTypes.CFuncTypeArg('it', PyrexTypes.py_object_type, None)])

    def _handle_simple_function_list(self, node, function, pos_args):
        """Turn list(ob) into PySequence_List(ob).
        """
        if len(pos_args) != 1:
            return node
        arg = pos_args[0]
        return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PySequence_ListKeepNew' if node.is_temp and arg.is_temp and (arg.type in (PyrexTypes.py_object_type, Builtin.list_type)) else 'PySequence_List', self.PySequence_List_func_type, args=pos_args, is_temp=node.is_temp)
    PyList_AsTuple_func_type = PyrexTypes.CFuncType(Builtin.tuple_type, [PyrexTypes.CFuncTypeArg('list', Builtin.list_type, None)])

    def _handle_simple_function_tuple(self, node, function, pos_args):
        """Replace tuple([...]) by PyList_AsTuple or PySequence_Tuple.
        """
        if len(pos_args) != 1 or not node.is_temp:
            return node
        arg = pos_args[0]
        if arg.type is Builtin.tuple_type and (not arg.may_be_none()):
            return arg
        if arg.type is Builtin.list_type:
            pos_args[0] = arg.as_none_safe_node("'NoneType' object is not iterable")
            return ExprNodes.PythonCapiCallNode(node.pos, 'PyList_AsTuple', self.PyList_AsTuple_func_type, args=pos_args, is_temp=node.is_temp)
        else:
            return ExprNodes.AsTupleNode(node.pos, arg=arg, type=Builtin.tuple_type)
    PySet_New_func_type = PyrexTypes.CFuncType(Builtin.set_type, [PyrexTypes.CFuncTypeArg('it', PyrexTypes.py_object_type, None)])

    def _handle_simple_function_set(self, node, function, pos_args):
        if len(pos_args) != 1:
            return node
        if pos_args[0].is_sequence_constructor:
            args = []
            temps = []
            for arg in pos_args[0].args:
                if not arg.is_simple():
                    arg = UtilNodes.LetRefNode(arg)
                    temps.append(arg)
                args.append(arg)
            result = ExprNodes.SetNode(node.pos, is_temp=1, args=args)
            self.replace(node, result)
            for temp in temps[::-1]:
                result = UtilNodes.EvalWithTempExprNode(temp, result)
            return result
        else:
            return self.replace(node, ExprNodes.PythonCapiCallNode(node.pos, 'PySet_New', self.PySet_New_func_type, args=pos_args, is_temp=node.is_temp, py_name='set'))
    PyFrozenSet_New_func_type = PyrexTypes.CFuncType(Builtin.frozenset_type, [PyrexTypes.CFuncTypeArg('it', PyrexTypes.py_object_type, None)])

    def _handle_simple_function_frozenset(self, node, function, pos_args):
        if not pos_args:
            pos_args = [ExprNodes.NullNode(node.pos)]
        elif len(pos_args) > 1:
            return node
        elif pos_args[0].type is Builtin.frozenset_type and (not pos_args[0].may_be_none()):
            return pos_args[0]
        return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PyFrozenSet_New', self.PyFrozenSet_New_func_type, args=pos_args, is_temp=node.is_temp, utility_code=UtilityCode.load_cached('pyfrozenset_new', 'Builtins.c'), py_name='frozenset')
    PyObject_AsDouble_func_type = PyrexTypes.CFuncType(PyrexTypes.c_double_type, [PyrexTypes.CFuncTypeArg('obj', PyrexTypes.py_object_type, None)], exception_value='((double)-1)', exception_check=True)

    def _handle_simple_function_float(self, node, function, pos_args):
        """Transform float() into either a C type cast or a faster C
        function call.
        """
        if len(pos_args) == 0:
            return ExprNodes.FloatNode(node, value='0.0', constant_result=0.0).coerce_to(Builtin.float_type, self.current_env())
        elif len(pos_args) != 1:
            self._error_wrong_arg_count('float', node, pos_args, '0 or 1')
            return node
        func_arg = pos_args[0]
        if isinstance(func_arg, ExprNodes.CoerceToPyTypeNode):
            func_arg = func_arg.arg
        if func_arg.type is PyrexTypes.c_double_type:
            return func_arg
        elif func_arg.type in (PyrexTypes.c_py_ucs4_type, PyrexTypes.c_py_unicode_type):
            return self._pyucs4_to_number(node, function.name, func_arg)
        elif node.type.assignable_from(func_arg.type) or func_arg.type.is_numeric:
            return ExprNodes.TypecastNode(node.pos, operand=func_arg, type=node.type)
        arg = pos_args[0].as_none_safe_node("float() argument must be a string or a number, not 'NoneType'")
        if func_arg.type is Builtin.bytes_type:
            cfunc_name = '__Pyx_PyBytes_AsDouble'
            utility_code_name = 'pybytes_as_double'
        elif func_arg.type is Builtin.bytearray_type:
            cfunc_name = '__Pyx_PyByteArray_AsDouble'
            utility_code_name = 'pybytes_as_double'
        elif func_arg.type is Builtin.unicode_type:
            cfunc_name = '__Pyx_PyUnicode_AsDouble'
            utility_code_name = 'pyunicode_as_double'
        elif func_arg.type is Builtin.str_type:
            cfunc_name = '__Pyx_PyString_AsDouble'
            utility_code_name = 'pystring_as_double'
        elif func_arg.type is Builtin.long_type:
            cfunc_name = 'PyLong_AsDouble'
        else:
            arg = pos_args[0]
            cfunc_name = '__Pyx_PyObject_AsDouble'
            utility_code_name = 'pyobject_as_double'
        return ExprNodes.PythonCapiCallNode(node.pos, cfunc_name, self.PyObject_AsDouble_func_type, args=[arg], is_temp=node.is_temp, utility_code=load_c_utility(utility_code_name) if utility_code_name else None, py_name='float')
    PyNumber_Int_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('o', PyrexTypes.py_object_type, None)])
    PyInt_FromDouble_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('value', PyrexTypes.c_double_type, None)])

    def _handle_simple_function_int(self, node, function, pos_args):
        """Transform int() into a faster C function call.
        """
        if len(pos_args) == 0:
            return ExprNodes.IntNode(node.pos, value='0', constant_result=0, type=PyrexTypes.py_object_type)
        elif len(pos_args) != 1:
            return node
        func_arg = pos_args[0]
        if isinstance(func_arg, ExprNodes.CoerceToPyTypeNode):
            if func_arg.arg.type.is_float:
                return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PyInt_FromDouble', self.PyInt_FromDouble_func_type, args=[func_arg.arg], is_temp=True, py_name='int', utility_code=UtilityCode.load_cached('PyIntFromDouble', 'TypeConversion.c'))
            else:
                return node
        if func_arg.type.is_pyobject and node.type.is_pyobject:
            return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PyNumber_Int', self.PyNumber_Int_func_type, args=pos_args, is_temp=True, py_name='int')
        return node

    def _handle_simple_function_bool(self, node, function, pos_args):
        """Transform bool(x) into a type coercion to a boolean.
        """
        if len(pos_args) == 0:
            return ExprNodes.BoolNode(node.pos, value=False, constant_result=False).coerce_to(Builtin.bool_type, self.current_env())
        elif len(pos_args) != 1:
            self._error_wrong_arg_count('bool', node, pos_args, '0 or 1')
            return node
        else:
            operand = pos_args[0].coerce_to_boolean(self.current_env())
            operand = ExprNodes.NotNode(node.pos, operand=operand)
            operand = ExprNodes.NotNode(node.pos, operand=operand)
            return operand.coerce_to_pyobject(self.current_env())
    PyMemoryView_FromObject_func_type = PyrexTypes.CFuncType(Builtin.memoryview_type, [PyrexTypes.CFuncTypeArg('value', PyrexTypes.py_object_type, None)])
    PyMemoryView_FromBuffer_func_type = PyrexTypes.CFuncType(Builtin.memoryview_type, [PyrexTypes.CFuncTypeArg('value', Builtin.py_buffer_type, None)])

    def _handle_simple_function_memoryview(self, node, function, pos_args):
        if len(pos_args) != 1:
            self._error_wrong_arg_count('memoryview', node, pos_args, '1')
            return node
        elif pos_args[0].type.is_pyobject:
            return ExprNodes.PythonCapiCallNode(node.pos, 'PyMemoryView_FromObject', self.PyMemoryView_FromObject_func_type, args=[pos_args[0]], is_temp=node.is_temp, py_name='memoryview')
        elif pos_args[0].type.is_ptr and pos_args[0].base_type is Builtin.py_buffer_type:
            return ExprNodes.PythonCapiCallNode(node.pos, 'PyMemoryView_FromBuffer', self.PyMemoryView_FromBuffer_func_type, args=[pos_args[0]], is_temp=node.is_temp, py_name='memoryview')
        return node
    Pyx_ssize_strlen_func_type = PyrexTypes.CFuncType(PyrexTypes.c_py_ssize_t_type, [PyrexTypes.CFuncTypeArg('bytes', PyrexTypes.c_const_char_ptr_type, None)], exception_value='-1')
    Pyx_Py_UNICODE_strlen_func_type = PyrexTypes.CFuncType(PyrexTypes.c_py_ssize_t_type, [PyrexTypes.CFuncTypeArg('unicode', PyrexTypes.c_const_py_unicode_ptr_type, None)], exception_value='-1')
    PyObject_Size_func_type = PyrexTypes.CFuncType(PyrexTypes.c_py_ssize_t_type, [PyrexTypes.CFuncTypeArg('obj', PyrexTypes.py_object_type, None)], exception_value='-1')
    _map_to_capi_len_function = {Builtin.unicode_type: '__Pyx_PyUnicode_GET_LENGTH', Builtin.bytes_type: '__Pyx_PyBytes_GET_SIZE', Builtin.bytearray_type: '__Pyx_PyByteArray_GET_SIZE', Builtin.list_type: '__Pyx_PyList_GET_SIZE', Builtin.tuple_type: '__Pyx_PyTuple_GET_SIZE', Builtin.set_type: '__Pyx_PySet_GET_SIZE', Builtin.frozenset_type: '__Pyx_PySet_GET_SIZE', Builtin.dict_type: 'PyDict_Size'}.get
    _ext_types_with_pysize = {'cpython.array.array'}

    def _handle_simple_function_len(self, node, function, pos_args):
        """Replace len(char*) by the equivalent call to strlen(),
        len(Py_UNICODE) by the equivalent Py_UNICODE_strlen() and
        len(known_builtin_type) by an equivalent C-API call.
        """
        if len(pos_args) != 1:
            self._error_wrong_arg_count('len', node, pos_args, 1)
            return node
        arg = pos_args[0]
        if isinstance(arg, ExprNodes.CoerceToPyTypeNode):
            arg = arg.arg
        if arg.type.is_string:
            new_node = ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_ssize_strlen', self.Pyx_ssize_strlen_func_type, args=[arg], is_temp=node.is_temp)
        elif arg.type.is_pyunicode_ptr:
            new_node = ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_Py_UNICODE_ssize_strlen', self.Pyx_Py_UNICODE_strlen_func_type, args=[arg], is_temp=node.is_temp, utility_code=UtilityCode.load_cached('ssize_pyunicode_strlen', 'StringTools.c'))
        elif arg.type.is_memoryviewslice:
            func_type = PyrexTypes.CFuncType(PyrexTypes.c_py_ssize_t_type, [PyrexTypes.CFuncTypeArg('memoryviewslice', arg.type, None)], nogil=True)
            new_node = ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_MemoryView_Len', func_type, args=[arg], is_temp=node.is_temp)
        elif arg.type.is_pyobject:
            cfunc_name = self._map_to_capi_len_function(arg.type)
            if cfunc_name is None:
                arg_type = arg.type
                if (arg_type.is_extension_type or arg_type.is_builtin_type) and arg_type.entry.qualified_name in self._ext_types_with_pysize:
                    cfunc_name = 'Py_SIZE'
                else:
                    return node
            arg = arg.as_none_safe_node("object of type 'NoneType' has no len()")
            new_node = ExprNodes.PythonCapiCallNode(node.pos, cfunc_name, self.PyObject_Size_func_type, args=[arg], is_temp=node.is_temp)
        elif arg.type.is_unicode_char:
            return ExprNodes.IntNode(node.pos, value='1', constant_result=1, type=node.type)
        else:
            return node
        if node.type not in (PyrexTypes.c_size_t_type, PyrexTypes.c_py_ssize_t_type):
            new_node = new_node.coerce_to(node.type, self.current_env())
        return new_node
    Pyx_Type_func_type = PyrexTypes.CFuncType(Builtin.type_type, [PyrexTypes.CFuncTypeArg('object', PyrexTypes.py_object_type, None)])

    def _handle_simple_function_type(self, node, function, pos_args):
        """Replace type(o) by a macro call to Py_TYPE(o).
        """
        if len(pos_args) != 1:
            return node
        node = ExprNodes.PythonCapiCallNode(node.pos, 'Py_TYPE', self.Pyx_Type_func_type, args=pos_args, is_temp=False)
        return ExprNodes.CastNode(node, PyrexTypes.py_object_type)
    Py_type_check_func_type = PyrexTypes.CFuncType(PyrexTypes.c_bint_type, [PyrexTypes.CFuncTypeArg('arg', PyrexTypes.py_object_type, None)])

    def _handle_simple_function_isinstance(self, node, function, pos_args):
        """Replace isinstance() checks against builtin types by the
        corresponding C-API call.
        """
        if len(pos_args) != 2:
            return node
        arg, types = pos_args
        temps = []
        if isinstance(types, ExprNodes.TupleNode):
            types = types.args
            if len(types) == 1 and (not types[0].type is Builtin.type_type):
                return node
            if arg.is_attribute or not arg.is_simple():
                arg = UtilNodes.ResultRefNode(arg)
                temps.append(arg)
        elif types.type is Builtin.type_type:
            types = [types]
        else:
            return node
        tests = []
        test_nodes = []
        env = self.current_env()
        for test_type_node in types:
            builtin_type = None
            if test_type_node.is_name:
                if test_type_node.entry:
                    entry = env.lookup(test_type_node.entry.name)
                    if entry and entry.type and entry.type.is_builtin_type:
                        builtin_type = entry.type
            if builtin_type is Builtin.type_type:
                if entry.name != 'type' or not (entry.scope and entry.scope.is_builtin_scope):
                    builtin_type = None
            if builtin_type is not None:
                type_check_function = entry.type.type_check_function(exact=False)
                if type_check_function == '__Pyx_Py3Int_Check' and builtin_type is Builtin.int_type:
                    type_check_function = 'PyInt_Check'
                if type_check_function in tests:
                    continue
                tests.append(type_check_function)
                type_check_args = [arg]
            elif test_type_node.type is Builtin.type_type:
                type_check_function = '__Pyx_TypeCheck'
                type_check_args = [arg, test_type_node]
            else:
                if not test_type_node.is_literal:
                    test_type_node = UtilNodes.ResultRefNode(test_type_node)
                    temps.append(test_type_node)
                type_check_function = 'PyObject_IsInstance'
                type_check_args = [arg, test_type_node]
            test_nodes.append(ExprNodes.PythonCapiCallNode(test_type_node.pos, type_check_function, self.Py_type_check_func_type, args=type_check_args, is_temp=True))

        def join_with_or(a, b, make_binop_node=ExprNodes.binop_node):
            or_node = make_binop_node(node.pos, 'or', a, b)
            or_node.type = PyrexTypes.c_bint_type
            or_node.wrap_operands(env)
            return or_node
        test_node = reduce(join_with_or, test_nodes).coerce_to(node.type, env)
        for temp in temps[::-1]:
            test_node = UtilNodes.EvalWithTempExprNode(temp, test_node)
        return test_node

    def _handle_simple_function_ord(self, node, function, pos_args):
        """Unpack ord(Py_UNICODE) and ord('X').
        """
        if len(pos_args) != 1:
            return node
        arg = pos_args[0]
        if isinstance(arg, ExprNodes.CoerceToPyTypeNode):
            if arg.arg.type.is_unicode_char:
                return ExprNodes.TypecastNode(arg.pos, operand=arg.arg, type=PyrexTypes.c_long_type).coerce_to(node.type, self.current_env())
        elif isinstance(arg, ExprNodes.UnicodeNode):
            if len(arg.value) == 1:
                return ExprNodes.IntNode(arg.pos, type=PyrexTypes.c_int_type, value=str(ord(arg.value)), constant_result=ord(arg.value)).coerce_to(node.type, self.current_env())
        elif isinstance(arg, ExprNodes.StringNode):
            if arg.unicode_value and len(arg.unicode_value) == 1 and (ord(arg.unicode_value) <= 255):
                return ExprNodes.IntNode(arg.pos, type=PyrexTypes.c_int_type, value=str(ord(arg.unicode_value)), constant_result=ord(arg.unicode_value)).coerce_to(node.type, self.current_env())
        return node
    Pyx_tp_new_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('type', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('args', Builtin.tuple_type, None)])
    Pyx_tp_new_kwargs_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('type', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('args', Builtin.tuple_type, None), PyrexTypes.CFuncTypeArg('kwargs', Builtin.dict_type, None)])

    def _handle_any_slot__new__(self, node, function, args, is_unbound_method, kwargs=None):
        """Replace 'exttype.__new__(exttype, ...)' by a call to exttype->tp_new()
        """
        obj = function.obj
        if not is_unbound_method or len(args) < 1:
            return node
        type_arg = args[0]
        if not obj.is_name or not type_arg.is_name:
            return node
        if obj.type != Builtin.type_type or type_arg.type != Builtin.type_type:
            return node
        if not type_arg.type_entry or not obj.type_entry:
            if obj.name != type_arg.name:
                return node
        elif type_arg.type_entry != obj.type_entry:
            return node
        args_tuple = ExprNodes.TupleNode(node.pos, args=args[1:])
        args_tuple = args_tuple.analyse_types(self.current_env(), skip_children=True)
        if type_arg.type_entry:
            ext_type = type_arg.type_entry.type
            if ext_type.is_extension_type and ext_type.typeobj_cname and (ext_type.scope.global_scope() == self.current_env().global_scope()):
                tp_slot = TypeSlots.ConstructorSlot('tp_new', '__new__')
                slot_func_cname = TypeSlots.get_slot_function(ext_type.scope, tp_slot)
                if slot_func_cname:
                    cython_scope = self.context.cython_scope
                    PyTypeObjectPtr = PyrexTypes.CPtrType(cython_scope.lookup('PyTypeObject').type)
                    pyx_tp_new_kwargs_func_type = PyrexTypes.CFuncType(ext_type, [PyrexTypes.CFuncTypeArg('type', PyTypeObjectPtr, None), PyrexTypes.CFuncTypeArg('args', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('kwargs', PyrexTypes.py_object_type, None)])
                    type_arg = ExprNodes.CastNode(type_arg, PyTypeObjectPtr)
                    if not kwargs:
                        kwargs = ExprNodes.NullNode(node.pos, type=PyrexTypes.py_object_type)
                    return ExprNodes.PythonCapiCallNode(node.pos, slot_func_cname, pyx_tp_new_kwargs_func_type, args=[type_arg, args_tuple, kwargs], may_return_none=False, is_temp=True)
        else:
            type_arg = type_arg.as_none_safe_node('object.__new__(X): X is not a type object (NoneType)')
        utility_code = UtilityCode.load_cached('tp_new', 'ObjectHandling.c')
        if kwargs:
            return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_tp_new_kwargs', self.Pyx_tp_new_kwargs_func_type, args=[type_arg, args_tuple, kwargs], utility_code=utility_code, is_temp=node.is_temp)
        else:
            return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_tp_new', self.Pyx_tp_new_func_type, args=[type_arg, args_tuple], utility_code=utility_code, is_temp=node.is_temp)

    def _handle_any_slot__class__(self, node, function, args, is_unbound_method, kwargs=None):
        return node
    PyObject_Append_func_type = PyrexTypes.CFuncType(PyrexTypes.c_returncode_type, [PyrexTypes.CFuncTypeArg('list', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('item', PyrexTypes.py_object_type, None)], exception_value='-1')

    def _handle_simple_method_object_append(self, node, function, args, is_unbound_method):
        """Optimistic optimisation as X.append() is almost always
        referring to a list.
        """
        if len(args) != 2 or node.result_is_used or node.function.entry:
            return node
        return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PyObject_Append', self.PyObject_Append_func_type, args=args, may_return_none=False, is_temp=node.is_temp, result_is_used=False, utility_code=load_c_utility('append'))

    def _handle_simple_method_list_extend(self, node, function, args, is_unbound_method):
        """Replace list.extend([...]) for short sequence literals values by sequential appends
        to avoid creating an intermediate sequence argument.
        """
        if len(args) != 2:
            return node
        obj, value = args
        if not value.is_sequence_constructor:
            return node
        items = list(value.args)
        if value.mult_factor is not None or len(items) > 8:
            if False and isinstance(value, ExprNodes.ListNode):
                tuple_node = args[1].as_tuple().analyse_types(self.current_env(), skip_children=True)
                Visitor.recursively_replace_node(node, args[1], tuple_node)
            return node
        wrapped_obj = self._wrap_self_arg(obj, function, is_unbound_method, 'extend')
        if not items:
            wrapped_obj.result_is_used = node.result_is_used
            return wrapped_obj
        cloned_obj = obj = wrapped_obj
        if len(items) > 1 and (not obj.is_simple()):
            cloned_obj = UtilNodes.LetRefNode(obj)
        temps = []
        arg = items[-1]
        if not arg.is_simple():
            arg = UtilNodes.LetRefNode(arg)
            temps.append(arg)
        new_node = ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PyList_Append', self.PyObject_Append_func_type, args=[cloned_obj, arg], is_temp=True, utility_code=load_c_utility('ListAppend'))
        for arg in items[-2::-1]:
            if not arg.is_simple():
                arg = UtilNodes.LetRefNode(arg)
                temps.append(arg)
            new_node = ExprNodes.binop_node(node.pos, '|', ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_ListComp_Append', self.PyObject_Append_func_type, args=[cloned_obj, arg], py_name='extend', is_temp=True, utility_code=load_c_utility('ListCompAppend')), new_node, type=PyrexTypes.c_returncode_type)
        new_node.result_is_used = node.result_is_used
        if cloned_obj is not obj:
            temps.append(cloned_obj)
        for temp in temps:
            new_node = UtilNodes.EvalWithTempExprNode(temp, new_node)
            new_node.result_is_used = node.result_is_used
        return new_node
    PyByteArray_Append_func_type = PyrexTypes.CFuncType(PyrexTypes.c_returncode_type, [PyrexTypes.CFuncTypeArg('bytearray', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('value', PyrexTypes.c_int_type, None)], exception_value='-1')
    PyByteArray_AppendObject_func_type = PyrexTypes.CFuncType(PyrexTypes.c_returncode_type, [PyrexTypes.CFuncTypeArg('bytearray', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('value', PyrexTypes.py_object_type, None)], exception_value='-1')

    def _handle_simple_method_bytearray_append(self, node, function, args, is_unbound_method):
        if len(args) != 2:
            return node
        func_name = '__Pyx_PyByteArray_Append'
        func_type = self.PyByteArray_Append_func_type
        value = unwrap_coerced_node(args[1])
        if value.type.is_int or isinstance(value, ExprNodes.IntNode):
            value = value.coerce_to(PyrexTypes.c_int_type, self.current_env())
            utility_code = UtilityCode.load_cached('ByteArrayAppend', 'StringTools.c')
        elif value.is_string_literal:
            if not value.can_coerce_to_char_literal():
                return node
            value = value.coerce_to(PyrexTypes.c_char_type, self.current_env())
            utility_code = UtilityCode.load_cached('ByteArrayAppend', 'StringTools.c')
        elif value.type.is_pyobject:
            func_name = '__Pyx_PyByteArray_AppendObject'
            func_type = self.PyByteArray_AppendObject_func_type
            utility_code = UtilityCode.load_cached('ByteArrayAppendObject', 'StringTools.c')
        else:
            return node
        new_node = ExprNodes.PythonCapiCallNode(node.pos, func_name, func_type, args=[args[0], value], may_return_none=False, is_temp=node.is_temp, utility_code=utility_code)
        if node.result_is_used:
            new_node = new_node.coerce_to(node.type, self.current_env())
        return new_node
    PyObject_Pop_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('list', PyrexTypes.py_object_type, None)])
    PyObject_PopIndex_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('list', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('py_index', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('c_index', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('is_signed', PyrexTypes.c_int_type, None)], has_varargs=True)

    def _handle_simple_method_list_pop(self, node, function, args, is_unbound_method):
        return self._handle_simple_method_object_pop(node, function, args, is_unbound_method, is_list=True)

    def _handle_simple_method_object_pop(self, node, function, args, is_unbound_method, is_list=False):
        """Optimistic optimisation as X.pop([n]) is almost always
        referring to a list.
        """
        if not args:
            return node
        obj = args[0]
        if is_list:
            type_name = 'List'
            obj = obj.as_none_safe_node("'NoneType' object has no attribute '%.30s'", error='PyExc_AttributeError', format_args=['pop'])
        else:
            type_name = 'Object'
        if len(args) == 1:
            return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_Py%s_Pop' % type_name, self.PyObject_Pop_func_type, args=[obj], may_return_none=True, is_temp=node.is_temp, utility_code=load_c_utility('pop'))
        elif len(args) == 2:
            index = unwrap_coerced_node(args[1])
            py_index = ExprNodes.NoneNode(index.pos)
            orig_index_type = index.type
            if not index.type.is_int:
                if isinstance(index, ExprNodes.IntNode):
                    py_index = index.coerce_to_pyobject(self.current_env())
                    index = index.coerce_to(PyrexTypes.c_py_ssize_t_type, self.current_env())
                elif is_list:
                    if index.type.is_pyobject:
                        py_index = index.coerce_to_simple(self.current_env())
                        index = ExprNodes.CloneNode(py_index)
                    index = index.coerce_to(PyrexTypes.c_py_ssize_t_type, self.current_env())
                else:
                    return node
            elif not PyrexTypes.numeric_type_fits(index.type, PyrexTypes.c_py_ssize_t_type):
                return node
            elif isinstance(index, ExprNodes.IntNode):
                py_index = index.coerce_to_pyobject(self.current_env())
            if not orig_index_type.is_int:
                orig_index_type = index.type
            if not orig_index_type.create_to_py_utility_code(self.current_env()):
                return node
            convert_func = orig_index_type.to_py_function
            conversion_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('intval', orig_index_type, None)])
            return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_Py%s_PopIndex' % type_name, self.PyObject_PopIndex_func_type, args=[obj, py_index, index, ExprNodes.IntNode(index.pos, value=str(orig_index_type.signed and 1 or 0), constant_result=orig_index_type.signed and 1 or 0, type=PyrexTypes.c_int_type), ExprNodes.RawCNameExprNode(index.pos, PyrexTypes.c_void_type, orig_index_type.empty_declaration_code()), ExprNodes.RawCNameExprNode(index.pos, conversion_type, convert_func)], may_return_none=True, is_temp=node.is_temp, utility_code=load_c_utility('pop_index'))
        return node
    single_param_func_type = PyrexTypes.CFuncType(PyrexTypes.c_returncode_type, [PyrexTypes.CFuncTypeArg('obj', PyrexTypes.py_object_type, None)], exception_value='-1')

    def _handle_simple_method_list_sort(self, node, function, args, is_unbound_method):
        """Call PyList_Sort() instead of the 0-argument l.sort().
        """
        if len(args) != 1:
            return node
        return self._substitute_method_call(node, function, 'PyList_Sort', self.single_param_func_type, 'sort', is_unbound_method, args).coerce_to(node.type, self.current_env)
    Pyx_PyDict_GetItem_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('dict', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('key', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('default', PyrexTypes.py_object_type, None)])

    def _handle_simple_method_dict_get(self, node, function, args, is_unbound_method):
        """Replace dict.get() by a call to PyDict_GetItem().
        """
        if len(args) == 2:
            args.append(ExprNodes.NoneNode(node.pos))
        elif len(args) != 3:
            self._error_wrong_arg_count('dict.get', node, args, '2 or 3')
            return node
        return self._substitute_method_call(node, function, '__Pyx_PyDict_GetItemDefault', self.Pyx_PyDict_GetItem_func_type, 'get', is_unbound_method, args, may_return_none=True, utility_code=load_c_utility('dict_getitem_default'))
    Pyx_PyDict_SetDefault_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('dict', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('key', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('default', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('is_safe_type', PyrexTypes.c_int_type, None)])

    def _handle_simple_method_dict_setdefault(self, node, function, args, is_unbound_method):
        """Replace dict.setdefault() by calls to PyDict_GetItem() and PyDict_SetItem().
        """
        if len(args) == 2:
            args.append(ExprNodes.NoneNode(node.pos))
        elif len(args) != 3:
            self._error_wrong_arg_count('dict.setdefault', node, args, '2 or 3')
            return node
        key_type = args[1].type
        if key_type.is_builtin_type:
            is_safe_type = int(key_type.name in 'str bytes unicode float int long bool')
        elif key_type is PyrexTypes.py_object_type:
            is_safe_type = -1
        else:
            is_safe_type = 0
        args.append(ExprNodes.IntNode(node.pos, value=str(is_safe_type), constant_result=is_safe_type))
        return self._substitute_method_call(node, function, '__Pyx_PyDict_SetDefault', self.Pyx_PyDict_SetDefault_func_type, 'setdefault', is_unbound_method, args, may_return_none=True, utility_code=load_c_utility('dict_setdefault'))
    PyDict_Pop_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('dict', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('key', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('default', PyrexTypes.py_object_type, None)])

    def _handle_simple_method_dict_pop(self, node, function, args, is_unbound_method):
        """Replace dict.pop() by a call to _PyDict_Pop().
        """
        if len(args) == 2:
            args.append(ExprNodes.NullNode(node.pos))
        elif len(args) != 3:
            self._error_wrong_arg_count('dict.pop', node, args, '2 or 3')
            return node
        return self._substitute_method_call(node, function, '__Pyx_PyDict_Pop', self.PyDict_Pop_func_type, 'pop', is_unbound_method, args, may_return_none=True, utility_code=load_c_utility('py_dict_pop'))
    Pyx_BinopInt_func_types = dict((((ctype, ret_type), PyrexTypes.CFuncType(ret_type, [PyrexTypes.CFuncTypeArg('op1', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('op2', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('cval', ctype, None), PyrexTypes.CFuncTypeArg('inplace', PyrexTypes.c_bint_type, None), PyrexTypes.CFuncTypeArg('zerodiv_check', PyrexTypes.c_bint_type, None)], exception_value=None if ret_type.is_pyobject else ret_type.exception_value)) for ctype in (PyrexTypes.c_long_type, PyrexTypes.c_double_type) for ret_type in (PyrexTypes.py_object_type, PyrexTypes.c_bint_type)))

    def _handle_simple_method_object___add__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('Add', node, function, args, is_unbound_method)

    def _handle_simple_method_object___sub__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('Subtract', node, function, args, is_unbound_method)

    def _handle_simple_method_object___mul__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('Multiply', node, function, args, is_unbound_method)

    def _handle_simple_method_object___eq__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('Eq', node, function, args, is_unbound_method)

    def _handle_simple_method_object___ne__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('Ne', node, function, args, is_unbound_method)

    def _handle_simple_method_object___and__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('And', node, function, args, is_unbound_method)

    def _handle_simple_method_object___or__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('Or', node, function, args, is_unbound_method)

    def _handle_simple_method_object___xor__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('Xor', node, function, args, is_unbound_method)

    def _handle_simple_method_object___rshift__(self, node, function, args, is_unbound_method):
        if len(args) != 2 or not isinstance(args[1], ExprNodes.IntNode):
            return node
        if not args[1].has_constant_result() or not 1 <= args[1].constant_result <= 63:
            return node
        return self._optimise_num_binop('Rshift', node, function, args, is_unbound_method)

    def _handle_simple_method_object___lshift__(self, node, function, args, is_unbound_method):
        if len(args) != 2 or not isinstance(args[1], ExprNodes.IntNode):
            return node
        if not args[1].has_constant_result() or not 1 <= args[1].constant_result <= 63:
            return node
        return self._optimise_num_binop('Lshift', node, function, args, is_unbound_method)

    def _handle_simple_method_object___mod__(self, node, function, args, is_unbound_method):
        return self._optimise_num_div('Remainder', node, function, args, is_unbound_method)

    def _handle_simple_method_object___floordiv__(self, node, function, args, is_unbound_method):
        return self._optimise_num_div('FloorDivide', node, function, args, is_unbound_method)

    def _handle_simple_method_object___truediv__(self, node, function, args, is_unbound_method):
        return self._optimise_num_div('TrueDivide', node, function, args, is_unbound_method)

    def _handle_simple_method_object___div__(self, node, function, args, is_unbound_method):
        return self._optimise_num_div('Divide', node, function, args, is_unbound_method)

    def _optimise_num_div(self, operator, node, function, args, is_unbound_method):
        if len(args) != 2 or not args[1].has_constant_result() or args[1].constant_result == 0:
            return node
        if isinstance(args[1], ExprNodes.IntNode):
            if not -2 ** 30 <= args[1].constant_result <= 2 ** 30:
                return node
        elif isinstance(args[1], ExprNodes.FloatNode):
            if not -2 ** 53 <= args[1].constant_result <= 2 ** 53:
                return node
        else:
            return node
        return self._optimise_num_binop(operator, node, function, args, is_unbound_method)

    def _handle_simple_method_float___add__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('Add', node, function, args, is_unbound_method)

    def _handle_simple_method_float___sub__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('Subtract', node, function, args, is_unbound_method)

    def _handle_simple_method_float___truediv__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('TrueDivide', node, function, args, is_unbound_method)

    def _handle_simple_method_float___div__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('Divide', node, function, args, is_unbound_method)

    def _handle_simple_method_float___mod__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('Remainder', node, function, args, is_unbound_method)

    def _handle_simple_method_float___eq__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('Eq', node, function, args, is_unbound_method)

    def _handle_simple_method_float___ne__(self, node, function, args, is_unbound_method):
        return self._optimise_num_binop('Ne', node, function, args, is_unbound_method)

    def _optimise_num_binop(self, operator, node, function, args, is_unbound_method):
        """
        Optimise math operators for (likely) float or small integer operations.
        """
        if getattr(node, 'special_bool_cmp_function', None):
            return node
        if len(args) != 2:
            return node
        if node.type.is_pyobject:
            ret_type = PyrexTypes.py_object_type
        elif node.type is PyrexTypes.c_bint_type and operator in ('Eq', 'Ne'):
            ret_type = PyrexTypes.c_bint_type
        else:
            return node
        result = optimise_numeric_binop(operator, node, ret_type, args[0], args[1])
        if not result:
            return node
        func_cname, utility_code, extra_args, num_type = result
        args = list(args) + extra_args
        call_node = self._substitute_method_call(node, function, func_cname, self.Pyx_BinopInt_func_types[num_type, ret_type], '__%s__' % operator[:3].lower(), is_unbound_method, args, may_return_none=True, with_none_check=False, utility_code=utility_code)
        if node.type.is_pyobject and (not ret_type.is_pyobject):
            call_node = ExprNodes.CoerceToPyTypeNode(call_node, self.current_env(), node.type)
        return call_node
    PyUnicode_uchar_predicate_func_type = PyrexTypes.CFuncType(PyrexTypes.c_bint_type, [PyrexTypes.CFuncTypeArg('uchar', PyrexTypes.c_py_ucs4_type, None)])

    def _inject_unicode_predicate(self, node, function, args, is_unbound_method):
        if is_unbound_method or len(args) != 1:
            return node
        ustring = args[0]
        if not isinstance(ustring, ExprNodes.CoerceToPyTypeNode) or not ustring.arg.type.is_unicode_char:
            return node
        uchar = ustring.arg
        method_name = function.attribute
        if method_name == 'istitle':
            utility_code = UtilityCode.load_cached('py_unicode_istitle', 'StringTools.c')
            function_name = '__Pyx_Py_UNICODE_ISTITLE'
        else:
            utility_code = None
            function_name = 'Py_UNICODE_%s' % method_name.upper()
        func_call = self._substitute_method_call(node, function, function_name, self.PyUnicode_uchar_predicate_func_type, method_name, is_unbound_method, [uchar], utility_code=utility_code)
        if node.type.is_pyobject:
            func_call = func_call.coerce_to_pyobject(self.current_env)
        return func_call
    _handle_simple_method_unicode_isalnum = _inject_unicode_predicate
    _handle_simple_method_unicode_isalpha = _inject_unicode_predicate
    _handle_simple_method_unicode_isdecimal = _inject_unicode_predicate
    _handle_simple_method_unicode_isdigit = _inject_unicode_predicate
    _handle_simple_method_unicode_islower = _inject_unicode_predicate
    _handle_simple_method_unicode_isnumeric = _inject_unicode_predicate
    _handle_simple_method_unicode_isspace = _inject_unicode_predicate
    _handle_simple_method_unicode_istitle = _inject_unicode_predicate
    _handle_simple_method_unicode_isupper = _inject_unicode_predicate
    PyUnicode_uchar_conversion_func_type = PyrexTypes.CFuncType(PyrexTypes.c_py_ucs4_type, [PyrexTypes.CFuncTypeArg('uchar', PyrexTypes.c_py_ucs4_type, None)])
    "\n    def _inject_unicode_character_conversion(self, node, function, args, is_unbound_method):\n        if is_unbound_method or len(args) != 1:\n            return node\n        ustring = args[0]\n        if not isinstance(ustring, ExprNodes.CoerceToPyTypeNode) or                not ustring.arg.type.is_unicode_char:\n            return node\n        uchar = ustring.arg\n        method_name = function.attribute\n        function_name = 'Py_UNICODE_TO%s' % method_name.upper()\n        func_call = self._substitute_method_call(\n            node, function,\n            function_name, self.PyUnicode_uchar_conversion_func_type,\n            method_name, is_unbound_method, [uchar])\n        if node.type.is_pyobject:\n            func_call = func_call.coerce_to_pyobject(self.current_env)\n        return func_call\n\n    #_handle_simple_method_unicode_lower = _inject_unicode_character_conversion\n    #_handle_simple_method_unicode_upper = _inject_unicode_character_conversion\n    #_handle_simple_method_unicode_title = _inject_unicode_character_conversion\n    "
    PyUnicode_Splitlines_func_type = PyrexTypes.CFuncType(Builtin.list_type, [PyrexTypes.CFuncTypeArg('str', Builtin.unicode_type, None), PyrexTypes.CFuncTypeArg('keepends', PyrexTypes.c_bint_type, None)])

    def _handle_simple_method_unicode_splitlines(self, node, function, args, is_unbound_method):
        """Replace unicode.splitlines(...) by a direct call to the
        corresponding C-API function.
        """
        if len(args) not in (1, 2):
            self._error_wrong_arg_count('unicode.splitlines', node, args, '1 or 2')
            return node
        self._inject_bint_default_argument(node, args, 1, False)
        return self._substitute_method_call(node, function, 'PyUnicode_Splitlines', self.PyUnicode_Splitlines_func_type, 'splitlines', is_unbound_method, args)
    PyUnicode_Split_func_type = PyrexTypes.CFuncType(Builtin.list_type, [PyrexTypes.CFuncTypeArg('str', Builtin.unicode_type, None), PyrexTypes.CFuncTypeArg('sep', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('maxsplit', PyrexTypes.c_py_ssize_t_type, None)])

    def _handle_simple_method_unicode_split(self, node, function, args, is_unbound_method):
        """Replace unicode.split(...) by a direct call to the
        corresponding C-API function.
        """
        if len(args) not in (1, 2, 3):
            self._error_wrong_arg_count('unicode.split', node, args, '1-3')
            return node
        if len(args) < 2:
            args.append(ExprNodes.NullNode(node.pos))
        else:
            self._inject_null_for_none(args, 1)
        self._inject_int_default_argument(node, args, 2, PyrexTypes.c_py_ssize_t_type, '-1')
        return self._substitute_method_call(node, function, 'PyUnicode_Split', self.PyUnicode_Split_func_type, 'split', is_unbound_method, args)
    PyUnicode_Join_func_type = PyrexTypes.CFuncType(Builtin.unicode_type, [PyrexTypes.CFuncTypeArg('str', Builtin.unicode_type, None), PyrexTypes.CFuncTypeArg('seq', PyrexTypes.py_object_type, None)])

    def _handle_simple_method_unicode_join(self, node, function, args, is_unbound_method):
        """
        unicode.join() builds a list first => see if we can do this more efficiently
        """
        if len(args) != 2:
            self._error_wrong_arg_count('unicode.join', node, args, '2')
            return node
        if isinstance(args[1], ExprNodes.GeneratorExpressionNode):
            gen_expr_node = args[1]
            loop_node = gen_expr_node.loop
            yield_statements = _find_yield_statements(loop_node)
            if yield_statements:
                inlined_genexpr = ExprNodes.InlinedGeneratorExpressionNode(node.pos, gen_expr_node, orig_func='list', comprehension_type=Builtin.list_type)
                for yield_expression, yield_stat_node in yield_statements:
                    append_node = ExprNodes.ComprehensionAppendNode(yield_expression.pos, expr=yield_expression, target=inlined_genexpr.target)
                    Visitor.recursively_replace_node(gen_expr_node, yield_stat_node, append_node)
                args[1] = inlined_genexpr
        return self._substitute_method_call(node, function, 'PyUnicode_Join', self.PyUnicode_Join_func_type, 'join', is_unbound_method, args)
    PyString_Tailmatch_func_type = PyrexTypes.CFuncType(PyrexTypes.c_bint_type, [PyrexTypes.CFuncTypeArg('str', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('substring', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('start', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('end', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('direction', PyrexTypes.c_int_type, None)], exception_value='-1')

    def _handle_simple_method_unicode_endswith(self, node, function, args, is_unbound_method):
        return self._inject_tailmatch(node, function, args, is_unbound_method, 'unicode', 'endswith', unicode_tailmatch_utility_code, +1)

    def _handle_simple_method_unicode_startswith(self, node, function, args, is_unbound_method):
        return self._inject_tailmatch(node, function, args, is_unbound_method, 'unicode', 'startswith', unicode_tailmatch_utility_code, -1)

    def _inject_tailmatch(self, node, function, args, is_unbound_method, type_name, method_name, utility_code, direction):
        """Replace unicode.startswith(...) and unicode.endswith(...)
        by a direct call to the corresponding C-API function.
        """
        if len(args) not in (2, 3, 4):
            self._error_wrong_arg_count('%s.%s' % (type_name, method_name), node, args, '2-4')
            return node
        self._inject_int_default_argument(node, args, 2, PyrexTypes.c_py_ssize_t_type, '0')
        self._inject_int_default_argument(node, args, 3, PyrexTypes.c_py_ssize_t_type, 'PY_SSIZE_T_MAX')
        args.append(ExprNodes.IntNode(node.pos, value=str(direction), type=PyrexTypes.c_int_type))
        method_call = self._substitute_method_call(node, function, '__Pyx_Py%s_Tailmatch' % type_name.capitalize(), self.PyString_Tailmatch_func_type, method_name, is_unbound_method, args, utility_code=utility_code)
        return method_call.coerce_to(Builtin.bool_type, self.current_env())
    PyUnicode_Find_func_type = PyrexTypes.CFuncType(PyrexTypes.c_py_ssize_t_type, [PyrexTypes.CFuncTypeArg('str', Builtin.unicode_type, None), PyrexTypes.CFuncTypeArg('substring', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('start', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('end', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('direction', PyrexTypes.c_int_type, None)], exception_value='-2')

    def _handle_simple_method_unicode_find(self, node, function, args, is_unbound_method):
        return self._inject_unicode_find(node, function, args, is_unbound_method, 'find', +1)

    def _handle_simple_method_unicode_rfind(self, node, function, args, is_unbound_method):
        return self._inject_unicode_find(node, function, args, is_unbound_method, 'rfind', -1)

    def _inject_unicode_find(self, node, function, args, is_unbound_method, method_name, direction):
        """Replace unicode.find(...) and unicode.rfind(...) by a
        direct call to the corresponding C-API function.
        """
        if len(args) not in (2, 3, 4):
            self._error_wrong_arg_count('unicode.%s' % method_name, node, args, '2-4')
            return node
        self._inject_int_default_argument(node, args, 2, PyrexTypes.c_py_ssize_t_type, '0')
        self._inject_int_default_argument(node, args, 3, PyrexTypes.c_py_ssize_t_type, 'PY_SSIZE_T_MAX')
        args.append(ExprNodes.IntNode(node.pos, value=str(direction), type=PyrexTypes.c_int_type))
        method_call = self._substitute_method_call(node, function, 'PyUnicode_Find', self.PyUnicode_Find_func_type, method_name, is_unbound_method, args)
        return method_call.coerce_to_pyobject(self.current_env())
    PyUnicode_Count_func_type = PyrexTypes.CFuncType(PyrexTypes.c_py_ssize_t_type, [PyrexTypes.CFuncTypeArg('str', Builtin.unicode_type, None), PyrexTypes.CFuncTypeArg('substring', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('start', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('end', PyrexTypes.c_py_ssize_t_type, None)], exception_value='-1')

    def _handle_simple_method_unicode_count(self, node, function, args, is_unbound_method):
        """Replace unicode.count(...) by a direct call to the
        corresponding C-API function.
        """
        if len(args) not in (2, 3, 4):
            self._error_wrong_arg_count('unicode.count', node, args, '2-4')
            return node
        self._inject_int_default_argument(node, args, 2, PyrexTypes.c_py_ssize_t_type, '0')
        self._inject_int_default_argument(node, args, 3, PyrexTypes.c_py_ssize_t_type, 'PY_SSIZE_T_MAX')
        method_call = self._substitute_method_call(node, function, 'PyUnicode_Count', self.PyUnicode_Count_func_type, 'count', is_unbound_method, args)
        return method_call.coerce_to_pyobject(self.current_env())
    PyUnicode_Replace_func_type = PyrexTypes.CFuncType(Builtin.unicode_type, [PyrexTypes.CFuncTypeArg('str', Builtin.unicode_type, None), PyrexTypes.CFuncTypeArg('substring', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('replstr', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('maxcount', PyrexTypes.c_py_ssize_t_type, None)])

    def _handle_simple_method_unicode_replace(self, node, function, args, is_unbound_method):
        """Replace unicode.replace(...) by a direct call to the
        corresponding C-API function.
        """
        if len(args) not in (3, 4):
            self._error_wrong_arg_count('unicode.replace', node, args, '3-4')
            return node
        self._inject_int_default_argument(node, args, 3, PyrexTypes.c_py_ssize_t_type, '-1')
        return self._substitute_method_call(node, function, 'PyUnicode_Replace', self.PyUnicode_Replace_func_type, 'replace', is_unbound_method, args)
    PyUnicode_AsEncodedString_func_type = PyrexTypes.CFuncType(Builtin.bytes_type, [PyrexTypes.CFuncTypeArg('obj', Builtin.unicode_type, None), PyrexTypes.CFuncTypeArg('encoding', PyrexTypes.c_const_char_ptr_type, None), PyrexTypes.CFuncTypeArg('errors', PyrexTypes.c_const_char_ptr_type, None)])
    PyUnicode_AsXyzString_func_type = PyrexTypes.CFuncType(Builtin.bytes_type, [PyrexTypes.CFuncTypeArg('obj', Builtin.unicode_type, None)])
    _special_encodings = ['UTF8', 'UTF16', 'UTF-16LE', 'UTF-16BE', 'Latin1', 'ASCII', 'unicode_escape', 'raw_unicode_escape']
    _special_codecs = [(name, codecs.getencoder(name)) for name in _special_encodings]

    def _handle_simple_method_unicode_encode(self, node, function, args, is_unbound_method):
        """Replace unicode.encode(...) by a direct C-API call to the
        corresponding codec.
        """
        if len(args) < 1 or len(args) > 3:
            self._error_wrong_arg_count('unicode.encode', node, args, '1-3')
            return node
        string_node = args[0]
        if len(args) == 1:
            null_node = ExprNodes.NullNode(node.pos)
            return self._substitute_method_call(node, function, 'PyUnicode_AsEncodedString', self.PyUnicode_AsEncodedString_func_type, 'encode', is_unbound_method, [string_node, null_node, null_node])
        parameters = self._unpack_encoding_and_error_mode(node.pos, args)
        if parameters is None:
            return node
        encoding, encoding_node, error_handling, error_handling_node = parameters
        if encoding and isinstance(string_node, ExprNodes.UnicodeNode):
            try:
                value = string_node.value.encode(encoding, error_handling)
            except:
                pass
            else:
                value = bytes_literal(value, encoding)
                return ExprNodes.BytesNode(string_node.pos, value=value, type=Builtin.bytes_type)
        if encoding and error_handling == 'strict':
            codec_name = self._find_special_codec_name(encoding)
            if codec_name is not None and '-' not in codec_name:
                encode_function = 'PyUnicode_As%sString' % codec_name
                return self._substitute_method_call(node, function, encode_function, self.PyUnicode_AsXyzString_func_type, 'encode', is_unbound_method, [string_node])
        return self._substitute_method_call(node, function, 'PyUnicode_AsEncodedString', self.PyUnicode_AsEncodedString_func_type, 'encode', is_unbound_method, [string_node, encoding_node, error_handling_node])
    PyUnicode_DecodeXyz_func_ptr_type = PyrexTypes.CPtrType(PyrexTypes.CFuncType(Builtin.unicode_type, [PyrexTypes.CFuncTypeArg('string', PyrexTypes.c_const_char_ptr_type, None), PyrexTypes.CFuncTypeArg('size', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('errors', PyrexTypes.c_const_char_ptr_type, None)]))
    _decode_c_string_func_type = PyrexTypes.CFuncType(Builtin.unicode_type, [PyrexTypes.CFuncTypeArg('string', PyrexTypes.c_const_char_ptr_type, None), PyrexTypes.CFuncTypeArg('start', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('stop', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('encoding', PyrexTypes.c_const_char_ptr_type, None), PyrexTypes.CFuncTypeArg('errors', PyrexTypes.c_const_char_ptr_type, None), PyrexTypes.CFuncTypeArg('decode_func', PyUnicode_DecodeXyz_func_ptr_type, None)])
    _decode_bytes_func_type = PyrexTypes.CFuncType(Builtin.unicode_type, [PyrexTypes.CFuncTypeArg('string', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('start', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('stop', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('encoding', PyrexTypes.c_const_char_ptr_type, None), PyrexTypes.CFuncTypeArg('errors', PyrexTypes.c_const_char_ptr_type, None), PyrexTypes.CFuncTypeArg('decode_func', PyUnicode_DecodeXyz_func_ptr_type, None)])
    _decode_cpp_string_func_type = None

    def _handle_simple_method_bytes_decode(self, node, function, args, is_unbound_method):
        """Replace char*.decode() by a direct C-API call to the
        corresponding codec, possibly resolving a slice on the char*.
        """
        if not 1 <= len(args) <= 3:
            self._error_wrong_arg_count('bytes.decode', node, args, '1-3')
            return node
        string_node = args[0]
        start = stop = None
        if isinstance(string_node, ExprNodes.SliceIndexNode):
            index_node = string_node
            string_node = index_node.base
            start, stop = (index_node.start, index_node.stop)
            if not start or start.constant_result == 0:
                start = None
        if isinstance(string_node, ExprNodes.CoerceToPyTypeNode):
            string_node = string_node.arg
        string_type = string_node.type
        if string_type in (Builtin.bytes_type, Builtin.bytearray_type):
            if is_unbound_method:
                string_node = string_node.as_none_safe_node("descriptor '%s' requires a '%s' object but received a 'NoneType'", format_args=['decode', string_type.name])
            else:
                string_node = string_node.as_none_safe_node("'NoneType' object has no attribute '%.30s'", error='PyExc_AttributeError', format_args=['decode'])
        elif not string_type.is_string and (not string_type.is_cpp_string):
            return node
        parameters = self._unpack_encoding_and_error_mode(node.pos, args)
        if parameters is None:
            return node
        encoding, encoding_node, error_handling, error_handling_node = parameters
        if not start:
            start = ExprNodes.IntNode(node.pos, value='0', constant_result=0)
        elif not start.type.is_int:
            start = start.coerce_to(PyrexTypes.c_py_ssize_t_type, self.current_env())
        if stop and (not stop.type.is_int):
            stop = stop.coerce_to(PyrexTypes.c_py_ssize_t_type, self.current_env())
        codec_name = None
        if encoding is not None:
            codec_name = self._find_special_codec_name(encoding)
        if codec_name is not None:
            if codec_name in ('UTF16', 'UTF-16LE', 'UTF-16BE'):
                codec_cname = '__Pyx_PyUnicode_Decode%s' % codec_name.replace('-', '')
            else:
                codec_cname = 'PyUnicode_Decode%s' % codec_name
            decode_function = ExprNodes.RawCNameExprNode(node.pos, type=self.PyUnicode_DecodeXyz_func_ptr_type, cname=codec_cname)
            encoding_node = ExprNodes.NullNode(node.pos)
        else:
            decode_function = ExprNodes.NullNode(node.pos)
        temps = []
        if string_type.is_string:
            if not stop:
                if not string_node.is_name:
                    string_node = UtilNodes.LetRefNode(string_node)
                    temps.append(string_node)
                stop = ExprNodes.PythonCapiCallNode(string_node.pos, '__Pyx_ssize_strlen', self.Pyx_ssize_strlen_func_type, args=[string_node], is_temp=True)
            helper_func_type = self._decode_c_string_func_type
            utility_code_name = 'decode_c_string'
        elif string_type.is_cpp_string:
            if not stop:
                stop = ExprNodes.IntNode(node.pos, value='PY_SSIZE_T_MAX', constant_result=ExprNodes.not_a_constant)
            if self._decode_cpp_string_func_type is None:
                self._decode_cpp_string_func_type = PyrexTypes.CFuncType(Builtin.unicode_type, [PyrexTypes.CFuncTypeArg('string', string_type, None), PyrexTypes.CFuncTypeArg('start', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('stop', PyrexTypes.c_py_ssize_t_type, None), PyrexTypes.CFuncTypeArg('encoding', PyrexTypes.c_const_char_ptr_type, None), PyrexTypes.CFuncTypeArg('errors', PyrexTypes.c_const_char_ptr_type, None), PyrexTypes.CFuncTypeArg('decode_func', self.PyUnicode_DecodeXyz_func_ptr_type, None)])
            helper_func_type = self._decode_cpp_string_func_type
            utility_code_name = 'decode_cpp_string'
        else:
            if not stop:
                stop = ExprNodes.IntNode(node.pos, value='PY_SSIZE_T_MAX', constant_result=ExprNodes.not_a_constant)
            helper_func_type = self._decode_bytes_func_type
            if string_type is Builtin.bytes_type:
                utility_code_name = 'decode_bytes'
            else:
                utility_code_name = 'decode_bytearray'
        node = ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_%s' % utility_code_name, helper_func_type, args=[string_node, start, stop, encoding_node, error_handling_node, decode_function], is_temp=node.is_temp, utility_code=UtilityCode.load_cached(utility_code_name, 'StringTools.c'))
        for temp in temps[::-1]:
            node = UtilNodes.EvalWithTempExprNode(temp, node)
        return node
    _handle_simple_method_bytearray_decode = _handle_simple_method_bytes_decode

    def _find_special_codec_name(self, encoding):
        try:
            requested_codec = codecs.getencoder(encoding)
        except LookupError:
            return None
        for name, codec in self._special_codecs:
            if codec == requested_codec:
                if '_' in name:
                    name = ''.join([s.capitalize() for s in name.split('_')])
                return name
        return None

    def _unpack_encoding_and_error_mode(self, pos, args):
        null_node = ExprNodes.NullNode(pos)
        if len(args) >= 2:
            encoding, encoding_node = self._unpack_string_and_cstring_node(args[1])
            if encoding_node is None:
                return None
        else:
            encoding = None
            encoding_node = null_node
        if len(args) == 3:
            error_handling, error_handling_node = self._unpack_string_and_cstring_node(args[2])
            if error_handling_node is None:
                return None
            if error_handling == 'strict':
                error_handling_node = null_node
        else:
            error_handling = 'strict'
            error_handling_node = null_node
        return (encoding, encoding_node, error_handling, error_handling_node)

    def _unpack_string_and_cstring_node(self, node):
        if isinstance(node, ExprNodes.CoerceToPyTypeNode):
            node = node.arg
        if isinstance(node, ExprNodes.UnicodeNode):
            encoding = node.value
            node = ExprNodes.BytesNode(node.pos, value=encoding.as_utf8_string(), type=PyrexTypes.c_const_char_ptr_type)
        elif isinstance(node, (ExprNodes.StringNode, ExprNodes.BytesNode)):
            encoding = node.value.decode('ISO-8859-1')
            node = ExprNodes.BytesNode(node.pos, value=node.value, type=PyrexTypes.c_const_char_ptr_type)
        elif node.type is Builtin.bytes_type:
            encoding = None
            node = node.coerce_to(PyrexTypes.c_const_char_ptr_type, self.current_env())
        elif node.type.is_string:
            encoding = None
        else:
            encoding = node = None
        return (encoding, node)

    def _handle_simple_method_str_endswith(self, node, function, args, is_unbound_method):
        return self._inject_tailmatch(node, function, args, is_unbound_method, 'str', 'endswith', str_tailmatch_utility_code, +1)

    def _handle_simple_method_str_startswith(self, node, function, args, is_unbound_method):
        return self._inject_tailmatch(node, function, args, is_unbound_method, 'str', 'startswith', str_tailmatch_utility_code, -1)

    def _handle_simple_method_bytes_endswith(self, node, function, args, is_unbound_method):
        return self._inject_tailmatch(node, function, args, is_unbound_method, 'bytes', 'endswith', bytes_tailmatch_utility_code, +1)

    def _handle_simple_method_bytes_startswith(self, node, function, args, is_unbound_method):
        return self._inject_tailmatch(node, function, args, is_unbound_method, 'bytes', 'startswith', bytes_tailmatch_utility_code, -1)
    "   # disabled for now, enable when we consider it worth it (see StringTools.c)\n    def _handle_simple_method_bytearray_endswith(self, node, function, args, is_unbound_method):\n        return self._inject_tailmatch(\n            node, function, args, is_unbound_method, 'bytearray', 'endswith',\n            bytes_tailmatch_utility_code, +1)\n\n    def _handle_simple_method_bytearray_startswith(self, node, function, args, is_unbound_method):\n        return self._inject_tailmatch(\n            node, function, args, is_unbound_method, 'bytearray', 'startswith',\n            bytes_tailmatch_utility_code, -1)\n    "

    def _substitute_method_call(self, node, function, name, func_type, attr_name, is_unbound_method, args=(), utility_code=None, is_temp=None, may_return_none=ExprNodes.PythonCapiCallNode.may_return_none, with_none_check=True):
        args = list(args)
        if with_none_check and args:
            args[0] = self._wrap_self_arg(args[0], function, is_unbound_method, attr_name)
        if is_temp is None:
            is_temp = node.is_temp
        return ExprNodes.PythonCapiCallNode(node.pos, name, func_type, args=args, is_temp=is_temp, utility_code=utility_code, may_return_none=may_return_none, result_is_used=node.result_is_used)

    def _wrap_self_arg(self, self_arg, function, is_unbound_method, attr_name):
        if self_arg.is_literal:
            return self_arg
        if is_unbound_method:
            self_arg = self_arg.as_none_safe_node("descriptor '%s' requires a '%s' object but received a 'NoneType'", format_args=[attr_name, self_arg.type.name])
        else:
            self_arg = self_arg.as_none_safe_node("'NoneType' object has no attribute '%{0}s'".format('.30' if len(attr_name) <= 30 else ''), error='PyExc_AttributeError', format_args=[attr_name])
        return self_arg
    obj_to_obj_func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('obj', PyrexTypes.py_object_type, None)])

    def _inject_null_for_none(self, args, index):
        if len(args) <= index:
            return
        arg = args[index]
        args[index] = ExprNodes.NullNode(arg.pos) if arg.is_none else ExprNodes.PythonCapiCallNode(arg.pos, '__Pyx_NoneAsNull', self.obj_to_obj_func_type, args=[arg.coerce_to_simple(self.current_env())], is_temp=0)

    def _inject_int_default_argument(self, node, args, arg_index, type, default_value):
        assert len(args) >= arg_index
        if len(args) == arg_index or args[arg_index].is_none:
            args.append(ExprNodes.IntNode(node.pos, value=str(default_value), type=type, constant_result=default_value))
        else:
            arg = args[arg_index].coerce_to(type, self.current_env())
            if isinstance(arg, ExprNodes.CoerceFromPyTypeNode):
                arg.special_none_cvalue = str(default_value)
            args[arg_index] = arg

    def _inject_bint_default_argument(self, node, args, arg_index, default_value):
        assert len(args) >= arg_index
        if len(args) == arg_index:
            default_value = bool(default_value)
            args.append(ExprNodes.BoolNode(node.pos, value=default_value, constant_result=default_value))
        else:
            args[arg_index] = args[arg_index].coerce_to_boolean(self.current_env())
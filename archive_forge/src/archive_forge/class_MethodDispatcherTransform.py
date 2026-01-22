from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
class MethodDispatcherTransform(EnvTransform):
    """
    Base class for transformations that want to intercept on specific
    builtin functions or methods of builtin types, including special
    methods triggered by Python operators.  Must run after declaration
    analysis when entries were assigned.

    Naming pattern for handler methods is as follows:

    * builtin functions: _handle_(general|simple|any)_function_NAME

    * builtin methods: _handle_(general|simple|any)_method_TYPENAME_METHODNAME
    """

    def visit_GeneralCallNode(self, node):
        self._process_children(node)
        function = node.function
        if not function.type.is_pyobject:
            return node
        arg_tuple = node.positional_args
        if not isinstance(arg_tuple, ExprNodes.TupleNode):
            return node
        keyword_args = node.keyword_args
        if keyword_args and (not isinstance(keyword_args, ExprNodes.DictNode)):
            return node
        args = arg_tuple.args
        return self._dispatch_to_handler(node, function, args, keyword_args)

    def visit_SimpleCallNode(self, node):
        self._process_children(node)
        function = node.function
        if function.type.is_pyobject:
            arg_tuple = node.arg_tuple
            if not isinstance(arg_tuple, ExprNodes.TupleNode):
                return node
            args = arg_tuple.args
        else:
            args = node.args
        return self._dispatch_to_handler(node, function, args, None)

    def visit_PrimaryCmpNode(self, node):
        if node.cascade:
            self._process_children(node)
            return node
        return self._visit_binop_node(node)

    def visit_BinopNode(self, node):
        return self._visit_binop_node(node)

    def _visit_binop_node(self, node):
        self._process_children(node)
        special_method_name = find_special_method_for_binary_operator(node.operator)
        if special_method_name:
            operand1, operand2 = (node.operand1, node.operand2)
            if special_method_name == '__contains__':
                operand1, operand2 = (operand2, operand1)
            elif special_method_name == '__div__':
                if Future.division in self.current_env().global_scope().context.future_directives:
                    special_method_name = '__truediv__'
            obj_type = operand1.type
            if obj_type.is_builtin_type:
                type_name = obj_type.name
            else:
                type_name = 'object'
            node = self._dispatch_to_method_handler(special_method_name, None, False, type_name, node, None, [operand1, operand2], None)
        return node

    def visit_UnopNode(self, node):
        self._process_children(node)
        special_method_name = find_special_method_for_unary_operator(node.operator)
        if special_method_name:
            operand = node.operand
            obj_type = operand.type
            if obj_type.is_builtin_type:
                type_name = obj_type.name
            else:
                type_name = 'object'
            node = self._dispatch_to_method_handler(special_method_name, None, False, type_name, node, None, [operand], None)
        return node

    def _find_handler(self, match_name, has_kwargs):
        try:
            match_name.encode('ascii')
        except UnicodeEncodeError:
            return None
        call_type = 'general' if has_kwargs else 'simple'
        handler = getattr(self, '_handle_%s_%s' % (call_type, match_name), None)
        if handler is None:
            handler = getattr(self, '_handle_any_%s' % match_name, None)
        return handler

    def _delegate_to_assigned_value(self, node, function, arg_list, kwargs):
        assignment = function.cf_state[0]
        value = assignment.rhs
        if value.is_name:
            if not value.entry or len(value.entry.cf_assignments) > 1:
                return node
        elif value.is_attribute and value.obj.is_name:
            if not value.obj.entry or len(value.obj.entry.cf_assignments) > 1:
                return node
        else:
            return node
        return self._dispatch_to_handler(node, value, arg_list, kwargs)

    def _dispatch_to_handler(self, node, function, arg_list, kwargs):
        if function.is_name:
            if not function.entry:
                return node
            entry = function.entry
            is_builtin = entry.is_builtin or entry is self.current_env().builtin_scope().lookup_here(function.name)
            if not is_builtin:
                if function.cf_state and function.cf_state.is_single:
                    return self._delegate_to_assigned_value(node, function, arg_list, kwargs)
                if arg_list and entry.is_cmethod and entry.scope and entry.scope.parent_type.is_builtin_type:
                    if entry.scope.parent_type is arg_list[0].type:
                        return self._dispatch_to_method_handler(entry.name, self_arg=None, is_unbound_method=True, type_name=entry.scope.parent_type.name, node=node, function=function, arg_list=arg_list, kwargs=kwargs)
                return node
            function_handler = self._find_handler('function_%s' % function.name, kwargs)
            if function_handler is None:
                return self._handle_function(node, function.name, function, arg_list, kwargs)
            if kwargs:
                return function_handler(node, function, arg_list, kwargs)
            else:
                return function_handler(node, function, arg_list)
        elif function.is_attribute:
            attr_name = function.attribute
            if function.type.is_pyobject:
                self_arg = function.obj
            elif node.self and function.entry:
                entry = function.entry.as_variable
                if not entry or not entry.is_builtin:
                    return node
                self_arg = node.self
                arg_list = arg_list[1:]
            else:
                return node
            obj_type = self_arg.type
            is_unbound_method = False
            if obj_type.is_builtin_type:
                if obj_type is Builtin.type_type and self_arg.is_name and arg_list and arg_list[0].type.is_pyobject:
                    type_name = self_arg.name
                    self_arg = None
                    is_unbound_method = True
                else:
                    type_name = obj_type.name
            else:
                type_name = 'object'
            return self._dispatch_to_method_handler(attr_name, self_arg, is_unbound_method, type_name, node, function, arg_list, kwargs)
        else:
            return node

    def _dispatch_to_method_handler(self, attr_name, self_arg, is_unbound_method, type_name, node, function, arg_list, kwargs):
        method_handler = self._find_handler('method_%s_%s' % (type_name, attr_name), kwargs)
        if method_handler is None:
            if attr_name in TypeSlots.special_method_names or attr_name in ['__new__', '__class__']:
                method_handler = self._find_handler('slot%s' % attr_name, kwargs)
            if method_handler is None:
                return self._handle_method(node, type_name, attr_name, function, arg_list, is_unbound_method, kwargs)
        if self_arg is not None:
            arg_list = [self_arg] + list(arg_list)
        if kwargs:
            result = method_handler(node, function, arg_list, is_unbound_method, kwargs)
        else:
            result = method_handler(node, function, arg_list, is_unbound_method)
        return result

    def _handle_function(self, node, function_name, function, arg_list, kwargs):
        """Fallback handler"""
        return node

    def _handle_method(self, node, type_name, attr_name, function, arg_list, is_unbound_method, kwargs):
        """Fallback handler"""
        return node
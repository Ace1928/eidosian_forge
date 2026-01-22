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
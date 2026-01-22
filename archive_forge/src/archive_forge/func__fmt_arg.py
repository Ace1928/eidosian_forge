from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
def _fmt_arg(self, arg):
    arg_doc = arg.name
    annotation = None
    defaultval = None
    if arg.is_self_arg:
        if self.is_format_clinic:
            arg_doc = '$self'
    elif arg.is_type_arg:
        if self.is_format_clinic:
            arg_doc = '$type'
    elif self.is_format_c:
        if arg.type is not PyrexTypes.py_object_type:
            arg_doc = arg.type.declaration_code(arg.name, for_display=1)
    elif self.is_format_python:
        if not arg.annotation:
            annotation = self._fmt_type(arg.type)
    if arg.annotation:
        if not self.is_format_clinic:
            annotation = self._fmt_annotation(arg.annotation)
    if arg.default:
        defaultval = self._fmt_expr(arg.default)
    if annotation:
        arg_doc = arg_doc + ': %s' % annotation
        if defaultval:
            arg_doc = arg_doc + ' = %s' % defaultval
    elif defaultval:
        arg_doc = arg_doc + '=%s' % defaultval
    return arg_doc
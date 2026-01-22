from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
def _fmt_star_arg(self, arg):
    arg_doc = arg.name
    if arg.annotation:
        if not self.is_format_clinic:
            annotation = self._fmt_annotation(arg.annotation)
            arg_doc = arg_doc + ': %s' % annotation
    return arg_doc
from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
def _setup_format(self):
    signature_format = self.current_directives['embedsignature.format']
    self.is_format_c = signature_format == 'c'
    self.is_format_python = signature_format == 'python'
    self.is_format_clinic = signature_format == 'clinic'
from . import (
from .Errors import error
from . import PyrexTypes
from .UtilityCode import CythonUtilityCode
from .Code import TempitaUtilityCode, UtilityCode
from .Visitor import PrintTree, TreeVisitor, VisitorTransform
def use_generic_utility_code(self):
    self.global_scope.use_utility_code(UtilityCode.load_cached('UFuncsInit', 'UFuncs_C.c'))
    self.global_scope.use_utility_code(UtilityCode.load_cached('NumpyImportUFunc', 'NumpyImportArray.c'))
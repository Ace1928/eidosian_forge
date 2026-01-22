from . import (
from .Errors import error
from . import PyrexTypes
from .UtilityCode import CythonUtilityCode
from .Code import TempitaUtilityCode, UtilityCode
from .Visitor import PrintTree, TreeVisitor, VisitorTransform
def _get_type_constant(pos, type_):
    if type_.is_complex:
        if type_ == PyrexTypes.c_float_complex_type:
            return 'NPY_CFLOAT'
        elif type_ == PyrexTypes.c_double_complex_type:
            return 'NPY_CDOUBLE'
        elif type_ == PyrexTypes.c_longdouble_complex_type:
            return 'NPY_CLONGDOUBLE'
    elif type_.is_numeric:
        postfix = type_.empty_declaration_code().upper().replace(' ', '')
        typename = 'NPY_%s' % postfix
        if typename in numpy_numeric_types:
            return typename
    elif type_.is_pyobject:
        return 'NPY_OBJECT'
    error(pos, "Type '%s' cannot be used as a ufunc argument" % type_)
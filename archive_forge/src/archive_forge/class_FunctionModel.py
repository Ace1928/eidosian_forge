from numba.extending import typeof_impl
from numba.extending import models, register_model
from numba.extending import unbox, NativeValue, box
from numba.core.imputils import lower_constant, lower_cast
from numba.core.ccallback import CFunc
from numba.core import cgutils
from llvmlite import ir
from numba.core import types
from numba.core.types import (FunctionType, UndefinedFunctionType,
from numba.core.dispatcher import Dispatcher
@register_model(FunctionType)
@register_model(UndefinedFunctionType)
class FunctionModel(models.StructModel):
    """FunctionModel holds addresses of function implementations
    """

    def __init__(self, dmm, fe_type):
        members = [('addr', types.voidptr), ('pyaddr', types.voidptr)]
        super(FunctionModel, self).__init__(dmm, fe_type, members)
from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.Opaque)
@register_default(types.PyObject)
@register_default(types.RawPointer)
@register_default(types.NoneType)
@register_default(types.StringLiteral)
@register_default(types.EllipsisType)
@register_default(types.Function)
@register_default(types.Type)
@register_default(types.Object)
@register_default(types.Module)
@register_default(types.Phantom)
@register_default(types.UndefVar)
@register_default(types.ContextManager)
@register_default(types.Dispatcher)
@register_default(types.ObjModeDispatcher)
@register_default(types.ExceptionClass)
@register_default(types.Dummy)
@register_default(types.ExceptionInstance)
@register_default(types.ExternalFunction)
@register_default(types.EnumClass)
@register_default(types.IntEnumClass)
@register_default(types.NumberClass)
@register_default(types.TypeRef)
@register_default(types.NamedTupleClass)
@register_default(types.DType)
@register_default(types.RecursiveCall)
@register_default(types.MakeFunctionLiteral)
@register_default(types.Poison)
class OpaqueModel(PrimitiveModel):
    """
    Passed as opaque pointers
    """
    _ptr_type = ir.IntType(8).as_pointer()

    def __init__(self, dmm, fe_type):
        be_type = self._ptr_type
        super(OpaqueModel, self).__init__(dmm, fe_type, be_type)
import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def test_parameterized_types(self):
    """https://github.com/numba/numba/issues/6401"""
    register_model(ParametrizedType)(UniTupleModel)

    @typeof_impl.register(Parametrized)
    def typeof_unit(val, c):
        return ParametrizedType(val)

    @unbox(ParametrizedType)
    def unbox_parametrized(typ, obj, context):
        return context.unbox(types.UniTuple(typ.dtype, len(typ)), obj)

    def dict_vs_cache_vs_parametrized(v):
        assert 0

    @overload(dict_vs_cache_vs_parametrized)
    def ol_dict_vs_cache_vs_parametrized(v):
        typ = v

        def objmode_vs_cache_vs_parametrized_impl(v):
            d = typed.Dict.empty(types.unicode_type, typ)
            d['data'] = v
        return objmode_vs_cache_vs_parametrized_impl

    @jit(nopython=True, cache=True)
    def set_parametrized_data(x, y):
        dict_vs_cache_vs_parametrized(x)
        dict_vs_cache_vs_parametrized(y)
    x, y = (Parametrized(('a', 'b')), Parametrized(('a',)))
    set_parametrized_data(x, y)
    set_parametrized_data._make_finalizer()()
    set_parametrized_data._reset_overloads()
    set_parametrized_data.targetctx.init()
    for ii in range(50):
        self.assertIsNone(set_parametrized_data(x, y))
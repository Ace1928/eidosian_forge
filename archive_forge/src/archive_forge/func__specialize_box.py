from functools import wraps, partial
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.decorators import njit
from numba.core.pythonapi import box, unbox, NativeValue
from numba.core.typing.typeof import typeof_impl
from numba.experimental.jitclass import _box
def _specialize_box(typ):
    """
    Create a subclass of Box that is specialized to the jitclass.

    This function caches the result to avoid code bloat.
    """
    if typ in _cache_specialized_box:
        return _cache_specialized_box[typ]
    dct = {'__slots__': (), '_numba_type_': typ, '__doc__': typ.class_type.class_doc}
    for field in typ.struct:
        getter = _generate_getter(field)
        setter = _generate_setter(field)
        dct[field] = property(getter, setter)
    for field, impdct in typ.jit_props.items():
        getter = None
        setter = None
        if 'get' in impdct:
            getter = _generate_getter(field)
        if 'set' in impdct:
            setter = _generate_setter(field)
        imp = impdct.get('get') or impdct.get('set') or None
        doc = getattr(imp, '__doc__', None)
        dct[field] = property(getter, setter, doc=doc)
    supported_dunders = {'__abs__', '__bool__', '__complex__', '__contains__', '__float__', '__getitem__', '__hash__', '__index__', '__int__', '__len__', '__setitem__', '__str__', '__eq__', '__ne__', '__ge__', '__gt__', '__le__', '__lt__', '__add__', '__floordiv__', '__lshift__', '__matmul__', '__mod__', '__mul__', '__neg__', '__pos__', '__pow__', '__rshift__', '__sub__', '__truediv__', '__and__', '__or__', '__xor__', '__iadd__', '__ifloordiv__', '__ilshift__', '__imatmul__', '__imod__', '__imul__', '__ipow__', '__irshift__', '__isub__', '__itruediv__', '__iand__', '__ior__', '__ixor__', '__radd__', '__rfloordiv__', '__rlshift__', '__rmatmul__', '__rmod__', '__rmul__', '__rpow__', '__rrshift__', '__rsub__', '__rtruediv__', '__rand__', '__ror__', '__rxor__'}
    for name, func in typ.methods.items():
        if name == '__init__':
            continue
        if name.startswith('__') and name.endswith('__') and (name not in supported_dunders):
            raise TypeError(f"Method '{name}' is not supported.")
        dct[name] = _generate_method(name, func)
    for name, func in typ.static_methods.items():
        dct[name] = _generate_method(name, func)
    subcls = type(typ.classname, (_box.Box,), dct)
    _cache_specialized_box[typ] = subcls
    for k, v in dct.items():
        if isinstance(v, property):
            prop = getattr(subcls, k)
            if prop.fget is not None:
                fget = prop.fget
                fast_fget = fget.compile((typ,))
                fget.disable_compile()
                setattr(subcls, k, property(fast_fget, prop.fset, prop.fdel, doc=prop.__doc__))
    return subcls
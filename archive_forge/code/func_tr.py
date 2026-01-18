import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def tr(t):

    def rec_tr(t, env):
        if isinstance(t, typing.TypeVar):
            if t in env:
                return env[t]
            else:
                env[t] = TypeVariable()
                return env[t]
        elif t is typing.Any:
            return TypeVariable()
        elif isinstance(t, NoneType_):
            return NoneType
        elif t is bool:
            return Bool()
        elif issubclass(t, slice):
            return Slice
        elif issubclass(t, (complex, complexfloating)):
            return Complex()
        elif issubclass(t, (float, floating)):
            return Float()
        elif issubclass(t, (int, integer)):
            return Integer()
        elif issubclass(t, NoneType_):
            return NoneType
        elif t is str:
            return Str()
        elif isinstance(t, typing.Generator):
            return Generator(rec_tr(t.__args__[0], env))
        elif isinstance(t, typing.List):
            return List(rec_tr(t.__args__[0], env))
        elif isinstance(t, typing.Optional):
            return OptionType(rec_tr(t.__args__[0], env))
        elif isinstance(t, typing.Set):
            return Set(rec_tr(t.__args__[0], env))
        elif isinstance(t, typing.Dict):
            return Dict(rec_tr(t.__args__[0], env), rec_tr(t.__args__[1], env))
        elif isinstance(t, typing.Tuple):
            return Tuple([rec_tr(tp, env) for tp in t.__args__])
        elif isinstance(t, typing.NDArray):
            return Array(rec_tr(t.__args__[0], env), len(t.__args__[1:]))
        elif isinstance(t, typing.Pointer):
            return Array(rec_tr(t.__args__[0], env), 1)
        elif isinstance(t, typing.Union):
            return MultiType([rec_tr(ut, env) for ut in t.__args__])
        elif t is typing.File:
            return File()
        elif isinstance(t, typing.Iterable):
            return Collection(TypeVariable(), TypeVariable(), TypeVariable(), rec_tr(t.__args__[0], env))
        elif t is typing.Sized:
            return Collection(Traits([TypeVariable(), LenTrait, TypeVariable()]), TypeVariable(), TypeVariable(), TypeVariable())
        elif isinstance(t, typing.Fun):
            return Function([rec_tr(at, env) for at in t.__args__[:-1]], rec_tr(t.__args__[-1], env))
        else:
            raise NotImplementedError(t)
    if isinstance(t, dict):
        return t
    elif hasattr(t, 'signature'):
        return rec_tr(t.signature, {})
    else:
        return rec_tr(t, {})
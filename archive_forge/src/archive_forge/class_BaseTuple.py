from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class BaseTuple(ConstSized, Hashable):
    """
    The base class for all tuple types (with a known size).
    """

    @classmethod
    def from_types(cls, tys, pyclass=None):
        """
        Instantiate the right tuple type for the given element types.
        """
        if pyclass is not None and pyclass is not tuple:
            assert issubclass(pyclass, tuple)
            if hasattr(pyclass, '_asdict'):
                tys = tuple(map(unliteral, tys))
                homogeneous = is_homogeneous(*tys)
                if homogeneous:
                    return NamedUniTuple(tys[0], len(tys), pyclass)
                else:
                    return NamedTuple(tys, pyclass)
        else:
            dtype = utils.unified_function_type(tys)
            if dtype is not None:
                return UniTuple(dtype, len(tys))
            homogeneous = is_homogeneous(*tys)
            if homogeneous:
                return cls._make_homogeneous_tuple(tys[0], len(tys))
            else:
                return cls._make_heterogeneous_tuple(tys)

    @classmethod
    def _make_homogeneous_tuple(cls, dtype, count):
        return UniTuple(dtype, count)

    @classmethod
    def _make_heterogeneous_tuple(cls, tys):
        return Tuple(tys)
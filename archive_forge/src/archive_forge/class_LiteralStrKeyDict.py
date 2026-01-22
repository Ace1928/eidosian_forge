from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class LiteralStrKeyDict(Literal, ConstSized, Hashable):
    """A Dictionary of string keys to heterogeneous values (basically a
    namedtuple with dict semantics).
    """

    class FakeNamedTuple(pySequence):

        def __init__(self, name, keys):
            self.__name__ = name
            self._fields = tuple(keys)
            super(LiteralStrKeyDict.FakeNamedTuple, self).__init__()

        def __len__(self):
            return len(self._fields)

        def __getitem__(self, key):
            return self._fields[key]
    mutable = False

    def __init__(self, literal_value, value_index=None):
        self._literal_init(literal_value)
        self.value_index = value_index
        strkeys = [x.literal_value for x in literal_value.keys()]
        self.tuple_ty = self.FakeNamedTuple('_ntclazz', strkeys)
        tys = [x for x in literal_value.values()]
        self.types = tuple(tys)
        self.count = len(self.types)
        self.fields = tuple(self.tuple_ty._fields)
        self.instance_class = self.tuple_ty
        self.name = 'LiteralStrKey[Dict]({})'.format(literal_value)

    def __unliteral__(self):
        return Poison(self)

    def unify(self, typingctx, other):
        """
        Unify this with the *other* one.
        """
        if isinstance(other, LiteralStrKeyDict):
            tys = []
            for (k1, v1), (k2, v2) in zip(self.literal_value.items(), other.literal_value.items()):
                if k1 != k2:
                    break
                tys.append(typingctx.unify_pairs(v1, v2))
            else:
                if all(tys):
                    d = {k: v for k, v in zip(self.literal_value.keys(), tys)}
                    return LiteralStrKeyDict(d)

    def __len__(self):
        return len(self.types)

    def __iter__(self):
        return iter(self.types)

    @property
    def key(self):
        return (self.tuple_ty._fields, self.types, str(self.literal_value))
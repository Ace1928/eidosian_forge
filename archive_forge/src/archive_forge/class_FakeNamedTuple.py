from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class FakeNamedTuple(pySequence):

    def __init__(self, name, keys):
        self.__name__ = name
        self._fields = tuple(keys)
        super(LiteralStrKeyDict.FakeNamedTuple, self).__init__()

    def __len__(self):
        return len(self._fields)

    def __getitem__(self, key):
        return self._fields[key]
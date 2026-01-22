from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class DictValuesIterableType(SimpleIterableType):
    """Dictionary iterable type for .values()
    """

    def __init__(self, parent):
        assert isinstance(parent, DictType)
        self.parent = parent
        self.yield_type = self.parent.value_type
        name = 'values[{}]'.format(self.parent.name)
        self.name = name
        iterator_type = DictIteratorType(self)
        super(DictValuesIterableType, self).__init__(name, iterator_type)
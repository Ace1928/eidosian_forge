from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class ListTypeIterableType(SimpleIterableType):
    """List iterable type
    """

    def __init__(self, parent):
        assert isinstance(parent, ListType)
        self.parent = parent
        self.yield_type = self.parent.item_type
        name = 'list[{}]'.format(self.parent.name)
        iterator_type = ListTypeIteratorType(self)
        super(ListTypeIterableType, self).__init__(name, iterator_type)
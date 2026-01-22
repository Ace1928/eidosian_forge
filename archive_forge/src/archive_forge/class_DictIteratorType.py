from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class DictIteratorType(SimpleIteratorType):

    def __init__(self, iterable):
        self.parent = iterable.parent
        self.iterable = iterable
        yield_type = iterable.yield_type
        name = 'iter[{}->{}],{}'.format(iterable.parent, yield_type, iterable.name)
        super(DictIteratorType, self).__init__(name, yield_type)
from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class BaseContainerPayload(Type):
    """
    Convenience base class for some container payloads.

    Derived classes must implement the *container_class* attribute.
    """

    def __init__(self, container):
        assert isinstance(container, self.container_class)
        self.container = container
        name = 'payload(%s)' % container
        super(BaseContainerPayload, self).__init__(name)

    @property
    def key(self):
        return self.container
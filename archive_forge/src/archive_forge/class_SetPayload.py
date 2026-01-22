from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class SetPayload(BaseContainerPayload):
    """
    Internal type class for the dynamically-allocated payload of a set.
    """
    container_class = Set
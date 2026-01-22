from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
def _sentry_forbidden_types(key, value):
    if isinstance(key, (Set, List)):
        raise TypingError('{} as key is forbidden'.format(key))
    if isinstance(value, (Set, List)):
        raise TypingError('{} as value is forbidden'.format(value))
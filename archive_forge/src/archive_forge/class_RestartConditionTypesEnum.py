from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
class RestartConditionTypesEnum:
    _values = ('none', 'on-failure', 'any')
    NONE, ON_FAILURE, ANY = _values
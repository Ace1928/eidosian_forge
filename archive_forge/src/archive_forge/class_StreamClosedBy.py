from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
class StreamClosedBy(Enum):
    SEND_END_STREAM = 0
    RECV_END_STREAM = 1
    SEND_RST_STREAM = 2
    RECV_RST_STREAM = 3
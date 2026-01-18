import errno as errno_mod
import zmq
from zmq.constants import SocketOption, _OptType
from zmq.error import ZMQError, _check_rc, _check_version
from ._cffi import ffi
from ._cffi import lib as C
from .message import Frame
from .utils import _retry_sys_call
def value_int64_pointer(val):
    return (ffi.new('int64_t*', val), ffi.sizeof('int64_t'))
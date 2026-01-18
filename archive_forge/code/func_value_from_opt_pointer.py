import errno as errno_mod
import zmq
from zmq.constants import SocketOption, _OptType
from zmq.error import ZMQError, _check_rc, _check_version
from ._cffi import ffi
from ._cffi import lib as C
from .message import Frame
from .utils import _retry_sys_call
def value_from_opt_pointer(option, opt_pointer, length=0):
    try:
        option = SocketOption(option)
    except ValueError:
        opt_type = _OptType.int
    else:
        opt_type = option._opt_type
    if opt_type == _OptType.bytes:
        return ffi.buffer(opt_pointer, length)[:]
    else:
        return int(opt_pointer[0])
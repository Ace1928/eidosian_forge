import warnings
from zmq.error import InterruptedSystemCall, _check_rc
from ._cffi import ffi
from ._cffi import lib as C
zmq poll function
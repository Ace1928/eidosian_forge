import errno
from threading import Event
import zmq
import zmq.error
from zmq.constants import ETERM
from ._cffi import ffi
from ._cffi import lib as C
def __buffer__(self, flags):
    return self.buffer
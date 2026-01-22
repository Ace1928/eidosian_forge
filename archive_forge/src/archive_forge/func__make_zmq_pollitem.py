import warnings
from zmq.error import InterruptedSystemCall, _check_rc
from ._cffi import ffi
from ._cffi import lib as C
def _make_zmq_pollitem(socket, flags):
    zmq_socket = socket._zmq_socket
    zmq_pollitem = ffi.new('zmq_pollitem_t*')
    zmq_pollitem.socket = zmq_socket
    zmq_pollitem.fd = 0
    zmq_pollitem.events = flags
    zmq_pollitem.revents = 0
    return zmq_pollitem[0]
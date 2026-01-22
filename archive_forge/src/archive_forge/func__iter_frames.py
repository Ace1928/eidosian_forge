import base64
import codecs
import collections
import errno
from random import Random
from socket import error as SocketError
import string
import struct
import sys
import time
import zlib
from eventlet import semaphore
from eventlet import wsgi
from eventlet.green import socket
from eventlet.support import get_errno
def _iter_frames(self):
    fragmented_message = None
    try:
        while True:
            message = self._recv_frame(message=fragmented_message)
            if message.opcode & 8:
                self._handle_control_frame(message.opcode, message.getvalue())
                continue
            if fragmented_message and message is not fragmented_message:
                raise RuntimeError('Unexpected message change.')
            fragmented_message = message
            if message.finished:
                data = fragmented_message.getvalue()
                fragmented_message = None
                yield data
    except FailedConnectionError:
        exc_typ, exc_val, exc_tb = sys.exc_info()
        self.close(close_data=(exc_val.status, exc_val.message))
    except ConnectionClosedError:
        return
    except Exception:
        self.close(close_data=(1011, 'Internal Server Error'))
        raise
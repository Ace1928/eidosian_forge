import errno
import io
import socket
from io import SEEK_END
from typing import Optional, Union
from ..exceptions import ConnectionError, TimeoutError
from ..utils import SSL_AVAILABLE
def unread_bytes(self) -> int:
    """
        Remaining unread length of buffer
        """
    pos = self._buffer.tell()
    end = self._buffer.seek(0, SEEK_END)
    self._buffer.seek(pos)
    return end - pos
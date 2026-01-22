from __future__ import annotations
from errno import EINTR
class ZMQError(ZMQBaseError):
    """Wrap an errno style error.

    Parameters
    ----------
    errno : int
        The ZMQ errno or None.  If None, then ``zmq_errno()`` is called and
        used.
    msg : str
        Description of the error or None.
    """
    errno: int | None = None

    def __init__(self, errno: int | None=None, msg: str | None=None):
        """Wrap an errno style error.

        Parameters
        ----------
        errno : int
            The ZMQ errno or None.  If None, then ``zmq_errno()`` is called and
            used.
        msg : string
            Description of the error or None.
        """
        from zmq.backend import strerror, zmq_errno
        if errno is None:
            errno = zmq_errno()
        if isinstance(errno, int):
            self.errno = errno
            if msg is None:
                self.strerror = strerror(errno)
            else:
                self.strerror = msg
        elif msg is None:
            self.strerror = str(errno)
        else:
            self.strerror = msg

    def __str__(self) -> str:
        return self.strerror

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{str(self)}')"
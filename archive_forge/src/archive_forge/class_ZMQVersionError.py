from __future__ import annotations
from errno import EINTR
class ZMQVersionError(NotImplementedError):
    """Raised when a feature is not provided by the linked version of libzmq.

    .. versionadded:: 14.2
    """
    min_version = None

    def __init__(self, min_version: str, msg: str='Feature'):
        global _zmq_version
        if _zmq_version is None:
            from zmq import zmq_version
            _zmq_version = zmq_version()
        self.msg = msg
        self.min_version = min_version
        self.version = _zmq_version

    def __repr__(self):
        return "ZMQVersionError('%s')" % str(self)

    def __str__(self):
        return f'{self.msg} requires libzmq >= {self.min_version}, have {self.version}'
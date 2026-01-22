import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
class AbortionError(Exception, metaclass=abc.ABCMeta):
    """Common super type for exceptions indicating RPC abortion.

    initial_metadata: The initial metadata from the other side of the RPC or
      None if no initial metadata value was received.
    terminal_metadata: The terminal metadata from the other side of the RPC or
      None if no terminal metadata value was received.
    code: The code value from the other side of the RPC or None if no code value
      was received.
    details: The details value from the other side of the RPC or None if no
      details value was received.
    """

    def __init__(self, initial_metadata, terminal_metadata, code, details):
        super(AbortionError, self).__init__()
        self.initial_metadata = initial_metadata
        self.terminal_metadata = terminal_metadata
        self.code = code
        self.details = details

    def __str__(self):
        return '%s(code=%s, details="%s")' % (self.__class__.__name__, self.code, self.details)
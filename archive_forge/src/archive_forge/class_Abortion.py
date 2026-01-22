import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
class Abortion(collections.namedtuple('Abortion', ('kind', 'initial_metadata', 'terminal_metadata', 'code', 'details'))):
    """A value describing RPC abortion.

    Attributes:
      kind: A Kind value identifying how the RPC failed.
      initial_metadata: The initial metadata from the other side of the RPC or
        None if no initial metadata value was received.
      terminal_metadata: The terminal metadata from the other side of the RPC or
        None if no terminal metadata value was received.
      code: The code value from the other side of the RPC or None if no code value
        was received.
      details: The details value from the other side of the RPC or None if no
        details value was received.
    """

    @enum.unique
    class Kind(enum.Enum):
        """Types of RPC abortion."""
        CANCELLED = 'cancelled'
        EXPIRED = 'expired'
        LOCAL_SHUTDOWN = 'local shutdown'
        REMOTE_SHUTDOWN = 'remote shutdown'
        NETWORK_FAILURE = 'network failure'
        LOCAL_FAILURE = 'local failure'
        REMOTE_FAILURE = 'remote failure'
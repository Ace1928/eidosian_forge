import abc
import enum
import threading  # pylint: disable=unused-import
class ProtocolReceiver(abc.ABC):
    """A means of receiving protocol values during an operation."""

    @abc.abstractmethod
    def context(self, protocol_context):
        """Accepts the protocol context object for the operation.

        Args:
          protocol_context: The protocol context object for the operation.
        """
        raise NotImplementedError()
import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
class RpcContext(abc.ABC):
    """Provides RPC-related information and action."""

    @abc.abstractmethod
    def is_active(self):
        """Describes whether the RPC is active or has terminated."""
        raise NotImplementedError()

    @abc.abstractmethod
    def time_remaining(self):
        """Describes the length of allowed time remaining for the RPC.

        Returns:
          A nonnegative float indicating the length of allowed time in seconds
          remaining for the RPC to complete before it is considered to have timed
          out.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def add_abortion_callback(self, abortion_callback):
        """Registers a callback to be called if the RPC is aborted.

        Args:
          abortion_callback: A callable to be called and passed an Abortion value
            in the event of RPC abortion.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def cancel(self):
        """Cancels the RPC.

        Idempotent and has no effect if the RPC has already terminated.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def protocol_context(self):
        """Accesses a custom object specified by an implementation provider.

        Returns:
          A value specified by the provider of a Face interface implementation
            affording custom state and behavior.
        """
        raise NotImplementedError()
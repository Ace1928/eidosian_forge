import abc
import grpc
class GRPCServicerContext(abc.ABC):
    """Exposes gRPC-specific options and behaviors to code servicing RPCs."""

    @abc.abstractmethod
    def peer(self):
        """Identifies the peer that invoked the RPC being serviced.

        Returns:
          A string identifying the peer that invoked the RPC being serviced.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def disable_next_response_compression(self):
        """Disables compression of the next response passed by the application."""
        raise NotImplementedError()
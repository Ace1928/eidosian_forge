import abc
import grpc
class GRPCInvocationContext(abc.ABC):
    """Exposes gRPC-specific options and behaviors to code invoking RPCs."""

    @abc.abstractmethod
    def disable_next_request_compression(self):
        """Disables compression of the next request passed by the application."""
        raise NotImplementedError()
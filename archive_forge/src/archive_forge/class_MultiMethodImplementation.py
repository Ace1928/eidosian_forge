import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
class MultiMethodImplementation(abc.ABC):
    """A general type able to service many methods."""

    @abc.abstractmethod
    def service(self, group, method, response_consumer, context):
        """Services an RPC.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          response_consumer: A stream.Consumer to be called to accept the response
            values of the RPC.
          context: a ServicerContext object.

        Returns:
          A stream.Consumer with which to accept the request values of the RPC. The
            consumer returned from this method may or may not be invoked to
            completion: in the case of RPC abortion, RPC Framework will simply stop
            passing values to this object. Implementations must not assume that this
            object will be called to completion of the request stream or even called
            at all.

        Raises:
          abandonment.Abandoned: May or may not be raised when the RPC has been
            aborted.
          NoSuchMethodError: If this MultiMethod does not recognize the given group
            and name for the RPC and is not able to service the RPC.
        """
        raise NotImplementedError()
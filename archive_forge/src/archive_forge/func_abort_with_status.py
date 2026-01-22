import abc
import contextlib
import enum
import logging
import sys
from grpc import _compression
from grpc._cython import cygrpc as _cygrpc
from grpc._runtime_protos import protos
from grpc._runtime_protos import protos_and_services
from grpc._runtime_protos import services
@abc.abstractmethod
def abort_with_status(self, status):
    """Raises an exception to terminate the RPC with a non-OK status.

        The status passed as argument will supercede any existing status code,
        status message and trailing metadata.

        This is an EXPERIMENTAL API.

        Args:
          status: A grpc.Status object. The status code in it must not be
            StatusCode.OK.

        Raises:
          Exception: An exception is always raised to signal the abortion the
            RPC to the gRPC runtime.
        """
    raise NotImplementedError()
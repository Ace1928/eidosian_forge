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
class AuthMetadataPluginCallback(abc.ABC):
    """Callback object received by a metadata plugin."""

    def __call__(self, metadata, error):
        """Passes to the gRPC runtime authentication metadata for an RPC.

        Args:
          metadata: The :term:`metadata` used to construct the CallCredentials.
          error: An Exception to indicate error or None to indicate success.
        """
        raise NotImplementedError()
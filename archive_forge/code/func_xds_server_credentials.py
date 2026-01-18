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
def xds_server_credentials(fallback_credentials):
    """Creates a ServerCredentials for use with xDS. This is an EXPERIMENTAL
      API.

    Args:
      fallback_credentials: Credentials to use in case it is not possible to
        establish a secure connection via xDS. No default value is provided.
    """
    return ServerCredentials(_cygrpc.xds_server_credentials(fallback_credentials._credentials))
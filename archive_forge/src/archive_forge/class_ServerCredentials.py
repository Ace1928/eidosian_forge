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
class ServerCredentials(object):
    """An encapsulation of the data required to open a secure port on a Server.

    This class has no supported interface - it exists to define the type of its
    instances and its instances exist to be passed to other functions.
    """

    def __init__(self, credentials):
        self._credentials = credentials
import asyncio
import sys
from typing import Any, Iterable, List, Optional, Sequence
import grpc
from grpc import _common
from grpc import _compression
from grpc import _grpcio_metadata
from grpc._cython import cygrpc
from . import _base_call
from . import _base_channel
from ._call import StreamStreamCall
from ._call import StreamUnaryCall
from ._call import UnaryStreamCall
from ._call import UnaryUnaryCall
from ._interceptor import ClientInterceptor
from ._interceptor import InterceptedStreamStreamCall
from ._interceptor import InterceptedStreamUnaryCall
from ._interceptor import InterceptedUnaryStreamCall
from ._interceptor import InterceptedUnaryUnaryCall
from ._interceptor import StreamStreamClientInterceptor
from ._interceptor import StreamUnaryClientInterceptor
from ._interceptor import UnaryStreamClientInterceptor
from ._interceptor import UnaryUnaryClientInterceptor
from ._metadata import Metadata
from ._typing import ChannelArgumentType
from ._typing import DeserializingFunction
from ._typing import MetadataType
from ._typing import RequestIterableType
from ._typing import RequestType
from ._typing import ResponseType
from ._typing import SerializingFunction
from ._utils import _timeout_to_deadline
Constructor.

        Args:
          target: The target to which to connect.
          options: Configuration options for the channel.
          credentials: A cygrpc.ChannelCredentials or None.
          compression: An optional value indicating the compression method to be
            used over the lifetime of the channel.
          interceptors: An optional list of interceptors that would be used for
            intercepting any RPC executed with that channel.
        
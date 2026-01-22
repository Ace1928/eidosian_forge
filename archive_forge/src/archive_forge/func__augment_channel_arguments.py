from concurrent.futures import Executor
from typing import Any, Optional, Sequence
import grpc
from grpc import _common
from grpc import _compression
from grpc._cython import cygrpc
from . import _base_server
from ._interceptor import ServerInterceptor
from ._typing import ChannelArgumentType
def _augment_channel_arguments(base_options: ChannelArgumentType, compression: Optional[grpc.Compression]):
    compression_option = _compression.create_channel_option(compression)
    return tuple(base_options) + compression_option
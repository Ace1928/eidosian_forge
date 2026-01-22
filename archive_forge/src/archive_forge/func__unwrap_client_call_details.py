import collections
import sys
import types
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import grpc
from ._typing import DeserializingFunction
from ._typing import DoneCallbackType
from ._typing import MetadataType
from ._typing import RequestIterableType
from ._typing import SerializingFunction
def _unwrap_client_call_details(call_details: grpc.ClientCallDetails, default_details: grpc.ClientCallDetails) -> Tuple[str, float, MetadataType, grpc.CallCredentials, bool, grpc.Compression]:
    try:
        method = call_details.method
    except AttributeError:
        method = default_details.method
    try:
        timeout = call_details.timeout
    except AttributeError:
        timeout = default_details.timeout
    try:
        metadata = call_details.metadata
    except AttributeError:
        metadata = default_details.metadata
    try:
        credentials = call_details.credentials
    except AttributeError:
        credentials = default_details.credentials
    try:
        wait_for_ready = call_details.wait_for_ready
    except AttributeError:
        wait_for_ready = default_details.wait_for_ready
    try:
        compression = call_details.compression
    except AttributeError:
        compression = default_details.compression
    return (method, timeout, metadata, credentials, wait_for_ready, compression)
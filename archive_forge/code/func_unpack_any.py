from typing import Dict, Tuple, TypeVar
from google.protobuf import any_pb2
from google.protobuf.message import Message
import cirq
def unpack_any(message: any_pb2.Any, out: M) -> M:
    message.Unpack(out)
    return out
from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import field_mask_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.speech_v1p1beta1.types import resource
class GetCustomClassRequest(proto.Message):
    """Message sent by the client for the ``GetCustomClass`` method.

    Attributes:
        name (str):
            Required. The name of the custom class to retrieve. Format:

            ``projects/{project}/locations/{location}/customClasses/{custom_class}``
    """
    name: str = proto.Field(proto.STRING, number=1)
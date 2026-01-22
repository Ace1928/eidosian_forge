from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import field_mask_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.speech_v1p1beta1.types import resource
class ListCustomClassesRequest(proto.Message):
    """Message sent by the client for the ``ListCustomClasses`` method.

    Attributes:
        parent (str):
            Required. The parent, which owns this collection of custom
            classes. Format:

            ``projects/{project}/locations/{location}/customClasses``

            Speech-to-Text supports three locations: ``global``, ``us``
            (US North America), and ``eu`` (Europe). If you are calling
            the ``speech.googleapis.com`` endpoint, use the ``global``
            location. To specify a region, use a `regional
            endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
            with matching ``us`` or ``eu`` location value.
        page_size (int):
            The maximum number of custom classes to
            return. The service may return fewer than this
            value. If unspecified, at most 50 custom classes
            will be returned. The maximum value is 1000;
            values above 1000 will be coerced to 1000.
        page_token (str):
            A page token, received from a previous ``ListCustomClass``
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            ``ListCustomClass`` must match the call that provided the
            page token.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
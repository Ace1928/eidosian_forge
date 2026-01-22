from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
class ListPhraseSetsRequest(proto.Message):
    """Request message for the
    [ListPhraseSets][google.cloud.speech.v2.Speech.ListPhraseSets]
    method.

    Attributes:
        parent (str):
            Required. The project and location of PhraseSet resources to
            list. The expected format is
            ``projects/{project}/locations/{location}``.
        page_size (int):
            The maximum number of PhraseSets to return.
            The service may return fewer than this value. If
            unspecified, at most 5 PhraseSets will be
            returned. The maximum value is 100; values above
            100 will be coerced to 100.
        page_token (str):
            A page token, received from a previous
            [ListPhraseSets][google.cloud.speech.v2.Speech.ListPhraseSets]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [ListPhraseSets][google.cloud.speech.v2.Speech.ListPhraseSets]
            must match the call that provided the page token.
        show_deleted (bool):
            Whether, or not, to show resources that have
            been deleted.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
    show_deleted: bool = proto.Field(proto.BOOL, number=4)
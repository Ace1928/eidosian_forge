from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
class SynthesisInput(proto.Message):
    """Contains text input to be synthesized. Either ``text`` or ``ssml``
    must be supplied. Supplying both or neither returns
    [google.rpc.Code.INVALID_ARGUMENT][google.rpc.Code.INVALID_ARGUMENT].
    The input size is limited to 5000 bytes.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        text (str):
            The raw text to be synthesized.

            This field is a member of `oneof`_ ``input_source``.
        ssml (str):
            The SSML document to be synthesized. The SSML document must
            be valid and well-formed. Otherwise the RPC will fail and
            return
            [google.rpc.Code.INVALID_ARGUMENT][google.rpc.Code.INVALID_ARGUMENT].
            For more information, see
            `SSML <https://cloud.google.com/text-to-speech/docs/ssml>`__.

            This field is a member of `oneof`_ ``input_source``.
    """
    text: str = proto.Field(proto.STRING, number=1, oneof='input_source')
    ssml: str = proto.Field(proto.STRING, number=2, oneof='input_source')
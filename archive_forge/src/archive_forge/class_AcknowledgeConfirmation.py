from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.pubsub_v1.types import schema as gp_schema
class AcknowledgeConfirmation(proto.Message):
    """Acknowledgement IDs sent in one or more previous requests to
        acknowledge a previously received message.

        Attributes:
            ack_ids (MutableSequence[str]):
                Successfully processed acknowledgement IDs.
            invalid_ack_ids (MutableSequence[str]):
                List of acknowledgement IDs that were
                malformed or whose acknowledgement deadline has
                expired.
            unordered_ack_ids (MutableSequence[str]):
                List of acknowledgement IDs that were out of
                order.
            temporary_failed_ack_ids (MutableSequence[str]):
                List of acknowledgement IDs that failed
                processing with temporary issues.
        """
    ack_ids: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=1)
    invalid_ack_ids: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=2)
    unordered_ack_ids: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=3)
    temporary_failed_ack_ids: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=4)
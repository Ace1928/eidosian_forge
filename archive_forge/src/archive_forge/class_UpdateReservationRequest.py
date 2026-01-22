from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class UpdateReservationRequest(proto.Message):
    """Request for UpdateReservation.

    Attributes:
        reservation (google.cloud.pubsublite_v1.types.Reservation):
            Required. The reservation to update. Its ``name`` field must
            be populated.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. A mask specifying the reservation
            fields to change.
    """
    reservation: common.Reservation = proto.Field(proto.MESSAGE, number=1, message=common.Reservation)
    update_mask: field_mask_pb2.FieldMask = proto.Field(proto.MESSAGE, number=2, message=field_mask_pb2.FieldMask)
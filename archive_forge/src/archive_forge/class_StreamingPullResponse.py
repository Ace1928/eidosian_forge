from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.pubsub_v1.types import schema as gp_schema
class StreamingPullResponse(proto.Message):
    """Response for the ``StreamingPull`` method. This response is used to
    stream messages from the server to the client.

    Attributes:
        received_messages (MutableSequence[google.pubsub_v1.types.ReceivedMessage]):
            Received Pub/Sub messages. This will not be
            empty.
        acknowledge_confirmation (google.pubsub_v1.types.StreamingPullResponse.AcknowledgeConfirmation):
            This field will only be set if
            ``enable_exactly_once_delivery`` is set to ``true``.
        modify_ack_deadline_confirmation (google.pubsub_v1.types.StreamingPullResponse.ModifyAckDeadlineConfirmation):
            This field will only be set if
            ``enable_exactly_once_delivery`` is set to ``true``.
        subscription_properties (google.pubsub_v1.types.StreamingPullResponse.SubscriptionProperties):
            Properties associated with this subscription.
    """

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

    class ModifyAckDeadlineConfirmation(proto.Message):
        """Acknowledgement IDs sent in one or more previous requests to
        modify the deadline for a specific message.

        Attributes:
            ack_ids (MutableSequence[str]):
                Successfully processed acknowledgement IDs.
            invalid_ack_ids (MutableSequence[str]):
                List of acknowledgement IDs that were
                malformed or whose acknowledgement deadline has
                expired.
            temporary_failed_ack_ids (MutableSequence[str]):
                List of acknowledgement IDs that failed
                processing with temporary issues.
        """
        ack_ids: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=1)
        invalid_ack_ids: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=2)
        temporary_failed_ack_ids: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=3)

    class SubscriptionProperties(proto.Message):
        """Subscription properties sent as part of the response.

        Attributes:
            exactly_once_delivery_enabled (bool):
                True iff exactly once delivery is enabled for
                this subscription.
            message_ordering_enabled (bool):
                True iff message ordering is enabled for this
                subscription.
        """
        exactly_once_delivery_enabled: bool = proto.Field(proto.BOOL, number=1)
        message_ordering_enabled: bool = proto.Field(proto.BOOL, number=2)
    received_messages: MutableSequence['ReceivedMessage'] = proto.RepeatedField(proto.MESSAGE, number=1, message='ReceivedMessage')
    acknowledge_confirmation: AcknowledgeConfirmation = proto.Field(proto.MESSAGE, number=5, message=AcknowledgeConfirmation)
    modify_ack_deadline_confirmation: ModifyAckDeadlineConfirmation = proto.Field(proto.MESSAGE, number=3, message=ModifyAckDeadlineConfirmation)
    subscription_properties: SubscriptionProperties = proto.Field(proto.MESSAGE, number=4, message=SubscriptionProperties)
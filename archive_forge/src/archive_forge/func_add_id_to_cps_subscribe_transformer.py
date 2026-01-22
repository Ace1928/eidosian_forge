import datetime
from google.api_core.exceptions import InvalidArgument
from cloudsdk.google.protobuf.timestamp_pb2 import Timestamp  # pytype: disable=pyi-error
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsublite.cloudpubsub import MessageTransformer
from google.cloud.pubsublite.internal import fast_serialize
from google.cloud.pubsublite.types import Partition, MessageMetadata
from google.cloud.pubsublite_v1 import AttributeValues, SequencedMessage, PubSubMessage
def add_id_to_cps_subscribe_transformer(partition: Partition, transformer: MessageTransformer) -> MessageTransformer:

    def add_id_to_message(source: SequencedMessage):
        source_pb = source._pb
        message: PubsubMessage = transformer.transform(source)
        message_pb = message._pb
        if message_pb.message_id:
            raise InvalidArgument('Message after transforming has the message_id field set.')
        message_pb.message_id = MessageMetadata._encode_parts(partition.value, source_pb.cursor.offset)
        return message
    return MessageTransformer.of_callable(add_id_to_message)
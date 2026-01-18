import datetime
from google.api_core.exceptions import InvalidArgument
from cloudsdk.google.protobuf.timestamp_pb2 import Timestamp  # pytype: disable=pyi-error
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsublite.cloudpubsub import MessageTransformer
from google.cloud.pubsublite.internal import fast_serialize
from google.cloud.pubsublite.types import Partition, MessageMetadata
from google.cloud.pubsublite_v1 import AttributeValues, SequencedMessage, PubSubMessage
def to_cps_subscribe_message(source: SequencedMessage) -> PubsubMessage:
    source_pb = source._pb
    out_pb = _to_cps_publish_message_proto(source_pb.message)
    out_pb.publish_time.CopyFrom(source_pb.publish_time)
    out = PubsubMessage()
    out._pb = out_pb
    return out
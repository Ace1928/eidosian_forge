import datetime
from google.api_core.exceptions import InvalidArgument
from cloudsdk.google.protobuf.timestamp_pb2 import Timestamp  # pytype: disable=pyi-error
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsublite.cloudpubsub import MessageTransformer
from google.cloud.pubsublite.internal import fast_serialize
from google.cloud.pubsublite.types import Partition, MessageMetadata
from google.cloud.pubsublite_v1 import AttributeValues, SequencedMessage, PubSubMessage
def _decode_attribute_event_time_proto(attr: str) -> Timestamp:
    try:
        ts = Timestamp()
        loaded = fast_serialize.load(attr)
        ts.seconds = loaded[0]
        ts.nanos = loaded[1]
        return ts
    except Exception:
        raise InvalidArgument('Invalid value for event time attribute.')
import asyncio
from typing import Callable, List, Dict, NamedTuple
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.pubsub_v1.subscriber.message import Message
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsublite.internal.wire.permanent_failable import adapt_error
from google.cloud.pubsublite.types import FlowControlSettings
from google.cloud.pubsublite.cloudpubsub.internal.ack_set_tracker import AckSetTracker
from google.cloud.pubsublite.cloudpubsub.internal.wrapped_message import (
from google.cloud.pubsublite.cloudpubsub.message_transformer import MessageTransformer
from google.cloud.pubsublite.cloudpubsub.nack_handler import NackHandler
from google.cloud.pubsublite.cloudpubsub.internal.single_subscriber import (
from google.cloud.pubsublite.internal.wire.permanent_failable import PermanentFailable
from google.cloud.pubsublite.internal.wire.subscriber import Subscriber
from google.cloud.pubsublite.internal.wire.subscriber_reset_handler import (
from google.cloud.pubsublite_v1 import FlowControlRequest, SequencedMessage
def _wrap_message(self, message: SequencedMessage.meta.pb) -> Message:
    rewrapped = SequencedMessage()
    rewrapped._pb = message
    cps_message = self._transformer.transform(rewrapped)
    offset = message.cursor.offset
    ack_id = AckId(self._ack_generation_id, offset)
    self._ack_set_tracker.track(offset)
    self._messages_by_ack_id[ack_id] = _SizedMessage(cps_message, message.size_bytes)
    wrapped_message = WrappedMessage(pb=cps_message._pb, ack_id=ack_id, ack_handler=lambda id, ack: self._on_ack_threadsafe(id, ack))
    return wrapped_message
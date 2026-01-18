from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional, Union, Set, AsyncIterator
from google.api_core.client_options import ClientOptions
from google.auth.credentials import Credentials
from google.cloud.pubsub_v1.subscriber.futures import StreamingPullFuture
from google.cloud.pubsub_v1.subscriber.message import Message
from google.cloud.pubsublite.cloudpubsub.reassignment_handler import ReassignmentHandler
from google.cloud.pubsublite.cloudpubsub.internal.make_subscriber import (
from google.cloud.pubsublite.cloudpubsub.internal.multiplexed_async_subscriber_client import (
from google.cloud.pubsublite.cloudpubsub.internal.multiplexed_subscriber_client import (
from google.cloud.pubsublite.cloudpubsub.message_transformer import MessageTransformer
from google.cloud.pubsublite.cloudpubsub.nack_handler import NackHandler
from google.cloud.pubsublite.cloudpubsub.subscriber_client_interface import (
from google.cloud.pubsublite.internal.constructable_from_service_account import (
from google.cloud.pubsublite.internal.require_started import RequireStarted
from google.cloud.pubsublite.types import (

        Create a new AsyncSubscriberClient.

        Args:
            nack_handler: A handler for when `nack()` is called. The default NackHandler raises an exception and fails the subscribe stream.
            message_transformer: A transformer from Pub/Sub Lite messages to Cloud Pub/Sub messages. This may not return a message with "message_id" set.
            credentials: If provided, the credentials to use when connecting.
            transport: The transport to use. Must correspond to an asyncio transport.
            client_options: The client options to use when connecting. If used, must explicitly set `api_endpoint`.
        
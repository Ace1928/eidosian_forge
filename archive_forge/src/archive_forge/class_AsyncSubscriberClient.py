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
class AsyncSubscriberClient(AsyncSubscriberClientInterface, ConstructableFromServiceAccount):
    """
    An AsyncSubscriberClient reads messages similar to Google Pub/Sub, but must be used in an
    async context.
    Any subscribe failures are unlikely to succeed if retried.

    Must be used in an `async with` block or have __aenter__() awaited before use.
    """
    _impl: AsyncSubscriberClientInterface
    _require_started: RequireStarted

    def __init__(self, *, nack_handler: Optional[NackHandler]=None, reassignment_handler: Optional[ReassignmentHandler]=None, message_transformer: Optional[MessageTransformer]=None, credentials: Optional[Credentials]=None, transport: str='grpc_asyncio', client_options: Optional[ClientOptions]=None):
        """
        Create a new AsyncSubscriberClient.

        Args:
            nack_handler: A handler for when `nack()` is called. The default NackHandler raises an exception and fails the subscribe stream.
            message_transformer: A transformer from Pub/Sub Lite messages to Cloud Pub/Sub messages. This may not return a message with "message_id" set.
            credentials: If provided, the credentials to use when connecting.
            transport: The transport to use. Must correspond to an asyncio transport.
            client_options: The client options to use when connecting. If used, must explicitly set `api_endpoint`.
        """
        self._impl = MultiplexedAsyncSubscriberClient(lambda subscription, partitions, settings: make_async_subscriber(subscription=subscription, transport=transport, per_partition_flow_control_settings=settings, nack_handler=nack_handler, reassignment_handler=reassignment_handler, message_transformer=message_transformer, fixed_partitions=partitions, credentials=credentials, client_options=client_options))
        self._require_started = RequireStarted()

    async def subscribe(self, subscription: Union[SubscriptionPath, str], per_partition_flow_control_settings: FlowControlSettings, fixed_partitions: Optional[Set[Partition]]=None) -> AsyncIterator[Message]:
        self._require_started.require_started()
        return await self._impl.subscribe(subscription, per_partition_flow_control_settings, fixed_partitions)

    async def __aenter__(self):
        self._require_started.__enter__()
        await self._impl.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self._impl.__aexit__(exc_type, exc_value, traceback)
        self._require_started.__exit__(exc_type, exc_value, traceback)
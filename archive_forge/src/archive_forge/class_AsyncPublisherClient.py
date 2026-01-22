from concurrent.futures import Future
from typing import Optional, Mapping, Union
from uuid import uuid4
from google.api_core.client_options import ClientOptions
from google.auth.credentials import Credentials
from google.cloud.pubsub_v1.types import BatchSettings
from google.cloud.pubsublite.cloudpubsub.internal.make_publisher import (
from google.cloud.pubsublite.cloudpubsub.internal.multiplexed_async_publisher_client import (
from google.cloud.pubsublite.cloudpubsub.internal.multiplexed_publisher_client import (
from google.cloud.pubsublite.cloudpubsub.publisher_client_interface import (
from google.cloud.pubsublite.internal.constructable_from_service_account import (
from google.cloud.pubsublite.internal.publisher_client_id import PublisherClientId
from google.cloud.pubsublite.internal.require_started import RequireStarted
from google.cloud.pubsublite.internal.wire.make_publisher import (
from google.cloud.pubsublite.types import TopicPath
class AsyncPublisherClient(AsyncPublisherClientInterface, ConstructableFromServiceAccount):
    """
    An AsyncPublisherClient publishes messages similar to Google Pub/Sub, but must be used in an
    async context. Any publish failures are unlikely to succeed if retried.

    Must be used in an `async with` block or have __aenter__() awaited before use.
    """
    _impl: AsyncPublisherClientInterface
    _require_started: RequireStarted
    DEFAULT_BATCHING_SETTINGS = WIRE_DEFAULT_BATCHING
    '\n    The default batching settings for a publisher client.\n    '

    def __init__(self, *, per_partition_batching_settings: Optional[BatchSettings]=None, credentials: Optional[Credentials]=None, transport: str='grpc_asyncio', client_options: Optional[ClientOptions]=None, enable_idempotence: bool=False):
        """
        Create a new AsyncPublisherClient.

        Args:
            per_partition_batching_settings: The settings for publish batching. Apply on a per-partition basis.
            credentials: If provided, the credentials to use when connecting.
            transport: The transport to use. Must correspond to an asyncio transport.
            client_options: The client options to use when connecting. If used, must explicitly set `api_endpoint`.
            enable_idempotence: Whether idempotence is enabled, where the server will ensure that unique messages within a single publisher session are stored only once.
        """
        client_id = _get_client_id(enable_idempotence)
        self._impl = MultiplexedAsyncPublisherClient(lambda topic: make_async_publisher(topic=topic, per_partition_batching_settings=per_partition_batching_settings, credentials=credentials, client_options=client_options, transport=transport, client_id=client_id))
        self._require_started = RequireStarted()

    async def publish(self, topic: Union[TopicPath, str], data: bytes, ordering_key: str='', **attrs: Mapping[str, str]) -> str:
        self._require_started.require_started()
        return await self._impl.publish(topic=topic, data=data, ordering_key=ordering_key, **attrs)

    async def __aenter__(self):
        self._require_started.__enter__()
        await self._impl.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self._impl.__aexit__(exc_type, exc_value, traceback)
        self._require_started.__exit__(exc_type, exc_value, traceback)
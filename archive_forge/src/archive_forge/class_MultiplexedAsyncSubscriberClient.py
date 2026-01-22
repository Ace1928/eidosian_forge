from typing import (
from google.cloud.pubsub_v1.subscriber.message import Message
from google.cloud.pubsublite.cloudpubsub.internal.single_subscriber import (
from google.cloud.pubsublite.cloudpubsub.subscriber_client_interface import (
from google.cloud.pubsublite.types import (
class MultiplexedAsyncSubscriberClient(AsyncSubscriberClientInterface):
    _underlying_factory: AsyncSubscriberFactory
    _live_clients: Set[AsyncSingleSubscriber]

    def __init__(self, underlying_factory: AsyncSubscriberFactory):
        self._underlying_factory = underlying_factory
        self._live_clients = set()

    async def subscribe(self, subscription: Union[SubscriptionPath, str], per_partition_flow_control_settings: FlowControlSettings, fixed_partitions: Optional[Set[Partition]]=None) -> AsyncIterator[Message]:
        if isinstance(subscription, str):
            subscription = SubscriptionPath.parse(subscription)
        subscriber = self._underlying_factory(subscription, fixed_partitions, per_partition_flow_control_settings)
        await subscriber.__aenter__()
        self._live_clients.add(subscriber)
        return _iterate_subscriber(subscriber, lambda: self._try_remove_client(subscriber))

    async def __aenter__(self):
        return self

    async def _try_remove_client(self, client: AsyncSingleSubscriber):
        if client in self._live_clients:
            self._live_clients.remove(client)
            await client.__aexit__(None, None, None)

    async def __aexit__(self, exc_type, exc_value, traceback):
        live_clients = self._live_clients
        self._live_clients = set()
        for client in live_clients:
            await client.__aexit__(None, None, None)
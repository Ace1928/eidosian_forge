from typing import Optional, Mapping, Set, AsyncIterator, Callable
from uuid import uuid4
from google.api_core.client_options import ClientOptions
from google.auth.credentials import Credentials
from google.cloud.pubsublite.cloudpubsub.reassignment_handler import (
from google.cloud.pubsublite.cloudpubsub.message_transforms import (
from google.cloud.pubsublite.internal.wire.client_cache import ClientCache
from google.cloud.pubsublite.types import FlowControlSettings
from google.cloud.pubsublite.cloudpubsub.internal.ack_set_tracker_impl import (
from google.cloud.pubsublite.cloudpubsub.internal.assigning_subscriber import (
from google.cloud.pubsublite.cloudpubsub.internal.single_partition_subscriber import (
from google.cloud.pubsublite.cloudpubsub.message_transformer import MessageTransformer
from google.cloud.pubsublite.cloudpubsub.nack_handler import (
from google.cloud.pubsublite.cloudpubsub.internal.single_subscriber import (
from google.cloud.pubsublite.internal.endpoints import regional_endpoint
from google.cloud.pubsublite.internal.wire.assigner import Assigner
from google.cloud.pubsublite.internal.wire.assigner_impl import AssignerImpl
from google.cloud.pubsublite.internal.wire.committer_impl import CommitterImpl
from google.cloud.pubsublite.internal.wire.fixed_set_assigner import FixedSetAssigner
from google.cloud.pubsublite.internal.wire.gapic_connection import (
from google.cloud.pubsublite.internal.wire.merge_metadata import merge_metadata
from google.cloud.pubsublite.internal.wire.pubsub_context import pubsub_context
import google.cloud.pubsublite.internal.wire.subscriber_impl as wire_subscriber
from google.cloud.pubsublite.internal.wire.subscriber_reset_handler import (
from google.cloud.pubsublite.types import Partition, SubscriptionPath
from google.cloud.pubsublite.internal.routing_metadata import (
from google.cloud.pubsublite_v1 import (
from google.cloud.pubsublite_v1.services.subscriber_service.async_client import (
from google.cloud.pubsublite_v1.services.partition_assignment_service.async_client import (
from google.cloud.pubsublite_v1.services.cursor_service.async_client import (
def _make_partition_subscriber_factory(subscription: SubscriptionPath, transport: str, client_options: ClientOptions, credentials: Optional[Credentials], base_metadata: Optional[Mapping[str, str]], flow_control_settings: FlowControlSettings, nack_handler: NackHandler, message_transformer: MessageTransformer) -> PartitionSubscriberFactory:
    subscribe_client_cache = ClientCache(lambda: SubscriberServiceAsyncClient(credentials=credentials, transport=transport, client_options=client_options))
    cursor_client_cache = ClientCache(lambda: CursorServiceAsyncClient(credentials=credentials, transport=transport, client_options=client_options))

    def factory(partition: Partition) -> AsyncSingleSubscriber:
        final_metadata = merge_metadata(base_metadata, subscription_routing_metadata(subscription, partition))

        def subscribe_connection_factory(requests: AsyncIterator[SubscribeRequest]):
            return subscribe_client_cache.get().subscribe(requests, metadata=list(final_metadata.items()))

        def cursor_connection_factory(requests: AsyncIterator[StreamingCommitCursorRequest]):
            return cursor_client_cache.get().streaming_commit_cursor(requests, metadata=list(final_metadata.items()))

        def subscriber_factory(reset_handler: SubscriberResetHandler):
            return wire_subscriber.SubscriberImpl(InitialSubscribeRequest(subscription=str(subscription), partition=partition.value), _DEFAULT_FLUSH_SECONDS, GapicConnectionFactory(subscribe_connection_factory), reset_handler)
        committer = CommitterImpl(InitialCommitCursorRequest(subscription=str(subscription), partition=partition.value), _DEFAULT_FLUSH_SECONDS, GapicConnectionFactory(cursor_connection_factory))
        ack_set_tracker = AckSetTrackerImpl(committer)
        return SinglePartitionSingleSubscriber(subscriber_factory, flow_control_settings, ack_set_tracker, nack_handler, add_id_to_cps_subscribe_transformer(partition, message_transformer))
    return factory
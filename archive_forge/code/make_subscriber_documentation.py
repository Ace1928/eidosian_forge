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

    Make a Pub/Sub Lite AsyncSubscriber.

    Args:
      subscription: The subscription to subscribe to.
      transport: The transport type to use.
      per_partition_flow_control_settings: The flow control settings for each partition subscribed to. Note that these
        settings apply to each partition individually, not in aggregate.
      nack_handler: An optional handler for when nack() is called on a Message. The default will fail the client.
      message_transformer: An optional transformer from Pub/Sub Lite messages to Cloud Pub/Sub messages.
      fixed_partitions: A fixed set of partitions to subscribe to. If not present, will instead use auto-assignment.
      credentials: The credentials to use to connect. GOOGLE_DEFAULT_CREDENTIALS is used if None.
      client_options: Other options to pass to the client. Note that if you pass any you must set api_endpoint.
      metadata: Additional metadata to send with the RPC.

    Returns:
      A new AsyncSubscriber.
    
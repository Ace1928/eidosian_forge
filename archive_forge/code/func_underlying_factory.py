from typing import Optional, Mapping
from google.api_core.client_options import ClientOptions
from google.auth.credentials import Credentials
from google.cloud.pubsub_v1.types import BatchSettings
from google.cloud.pubsublite.cloudpubsub.internal.async_publisher_impl import (
from google.cloud.pubsublite.cloudpubsub.internal.publisher_impl import (
from google.cloud.pubsublite.cloudpubsub.internal.single_publisher import (
from google.cloud.pubsublite.internal.publisher_client_id import PublisherClientId
from google.cloud.pubsublite.internal.wire.make_publisher import (
from google.cloud.pubsublite.internal.wire.merge_metadata import merge_metadata
from google.cloud.pubsublite.internal.wire.pubsub_context import pubsub_context
from google.cloud.pubsublite.types import TopicPath
def underlying_factory():
    return make_wire_publisher(topic=topic, transport=transport, per_partition_batching_settings=per_partition_batching_settings, credentials=credentials, client_options=client_options, metadata=metadata, client_id=client_id)
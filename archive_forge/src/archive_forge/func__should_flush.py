import asyncio
from typing import Optional, List, NamedTuple
import logging
from google.cloud.pubsub_v1.types import BatchSettings
from google.cloud.pubsublite.internal.publish_sequence_number import (
from google.cloud.pubsublite.internal.wait_ignore_cancelled import wait_ignore_errors
from google.cloud.pubsublite.internal.wire.publisher import Publisher
from google.cloud.pubsublite.internal.wire.retrying_connection import (
from google.api_core.exceptions import FailedPrecondition, GoogleAPICallError
from google.cloud.pubsublite.internal.wire.connection_reinitializer import (
from google.cloud.pubsublite.internal.wire.connection import Connection
from google.cloud.pubsublite.internal.wire.serial_batcher import (
from google.cloud.pubsublite.types import Partition, MessageMetadata
from google.cloud.pubsublite_v1.types import (
from google.cloud.pubsublite.internal.wire.work_item import WorkItem
def _should_flush(self) -> bool:
    size = self._batcher.size()
    return size.element_count >= self._batching_settings.max_messages or size.byte_count >= self._batching_settings.max_bytes
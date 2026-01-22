import logging
from concurrent.futures.thread import ThreadPoolExecutor
import asyncio
from google.cloud.pubsublite import AdminClientInterface
from google.cloud.pubsublite.internal.wait_ignore_cancelled import wait_ignore_cancelled
from google.cloud.pubsublite.internal.wire.partition_count_watcher import (
from google.cloud.pubsublite.internal.wire.permanent_failable import PermanentFailable
from google.cloud.pubsublite.types import TopicPath
from google.api_core.exceptions import GoogleAPICallError
def _get_partition_count_sync(self) -> int:
    return self._admin.get_topic_partition_count(self._topic_path)
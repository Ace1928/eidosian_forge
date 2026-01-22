from concurrent.futures.thread import ThreadPoolExecutor
from typing import Union, Optional, Set
from threading import Lock
from google.cloud.pubsub_v1.subscriber.futures import StreamingPullFuture
from google.cloud.pubsublite.cloudpubsub.internal.single_subscriber import (
from google.cloud.pubsublite.cloudpubsub.internal.subscriber_impl import SubscriberImpl
from google.cloud.pubsublite.cloudpubsub.subscriber_client_interface import (
from google.cloud.pubsublite.types import (
def _try_remove_client(self, future: StreamingPullFuture):
    with self._lock:
        if future not in self._live_clients:
            return
        self._live_clients.remove(future)
    self._cancel_streaming_pull_future(future)
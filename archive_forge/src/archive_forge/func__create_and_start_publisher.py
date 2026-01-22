from concurrent.futures import Future
from typing import Callable, Union, Mapping
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.pubsublite.cloudpubsub.internal.client_multiplexer import (
from google.cloud.pubsublite.cloudpubsub.internal.single_publisher import (
from google.cloud.pubsublite.cloudpubsub.publisher_client_interface import (
from google.cloud.pubsublite.types import TopicPath
def _create_and_start_publisher(self, topic: Union[TopicPath, str]):
    publisher = self._publisher_factory(topic)
    try:
        return publisher.__enter__()
    except GoogleAPICallError:
        publisher.__exit__(None, None, None)
        raise
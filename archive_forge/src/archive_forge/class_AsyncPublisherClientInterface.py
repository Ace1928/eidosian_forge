from abc import abstractmethod, ABCMeta
from concurrent.futures import Future
from typing import ContextManager, Mapping, Union, AsyncContextManager
from google.cloud.pubsublite.types import TopicPath
class AsyncPublisherClientInterface(AsyncContextManager, metaclass=ABCMeta):
    """
    An AsyncPublisherClientInterface publishes messages similar to Google Pub/Sub, but must be used in an
    async context. Any publish failures are unlikely to succeed if retried.

    Must be used in an `async with` block or have __aenter__() awaited before use.
    """

    @abstractmethod
    async def publish(self, topic: Union[TopicPath, str], data: bytes, ordering_key: str='', **attrs: Mapping[str, str]) -> str:
        """
        Publish a message.

        Args:
          topic: The topic to publish to. Publishes to new topics may have nontrivial startup latency.
          data: The bytestring payload of the message
          ordering_key: The key to enforce ordering on, or "" for no ordering.
          **attrs: Additional attributes to send.

        Returns:
          An ack id, which can be decoded using MessageMetadata.decode.

        Raises:
          GoogleApiCallError: On a permanent failure.
        """
        raise NotImplementedError()
from abc import abstractmethod, ABCMeta
from concurrent.futures import Future
from typing import ContextManager, Mapping, Union, AsyncContextManager
from google.cloud.pubsublite.types import TopicPath

        Publish a message.

        Args:
          topic: The topic to publish to. Publishes to new topics may have nontrivial startup latency.
          data: The bytestring payload of the message
          ordering_key: The key to enforce ordering on, or "" for no ordering.
          **attrs: Additional attributes to send.

        Returns:
          A future completed with an ack id of type str, which can be decoded using
          MessageMetadata.decode.

        Raises:
          GoogleApiCallError: On a permanent failure.
        
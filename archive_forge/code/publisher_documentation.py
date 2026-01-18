from abc import abstractmethod, ABCMeta
from typing import AsyncContextManager
from google.cloud.pubsublite_v1.types import PubSubMessage
from google.cloud.pubsublite.types import MessageMetadata

        Publish the provided message.

        Args:
          message: The message to be published.

        Returns:
          Metadata about the published message.

        Raises:
          GoogleAPICallError: On a permanent error.
        
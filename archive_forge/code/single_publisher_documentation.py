from abc import abstractmethod, ABCMeta
from typing import AsyncContextManager, Mapping, ContextManager
from concurrent import futures

        Publish a message.

        Args:
          data: The bytestring payload of the message
          ordering_key: The key to enforce ordering on, or "" for no ordering.
          **attrs: Additional attributes to send.

        Returns:
          A future completed with an ack id, which can be decoded using MessageMetadata.decode.

        Raises:
          GoogleApiCallError: On a permanent failure.
        
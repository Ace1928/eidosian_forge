from abc import abstractmethod, ABCMeta
from typing import AsyncContextManager, Callable, List, Set, Optional
from google.cloud.pubsub_v1.subscriber.message import Message
from google.cloud.pubsublite.types import (

        Read the next batch off of the stream.

        Returns:
          The next batch of messages. ack() or nack() must eventually be called
          exactly once on each message.

          Pub/Sub Lite does not support nack() by default- if you do call nack(), it will immediately fail the client
          unless you have a NackHandler installed.

        Raises:
          GoogleAPICallError: On a permanent error.
        
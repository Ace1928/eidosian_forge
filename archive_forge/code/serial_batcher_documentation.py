from abc import abstractmethod, ABCMeta
from typing import Generic, List, NamedTuple
import asyncio
from google.cloud.pubsublite.internal.wire.connection import Request, Response
from google.cloud.pubsublite.internal.wire.work_item import WorkItem
Add a new request to this batcher.

        Args:
          request: The request to send.

        Returns:
          A future that will resolve to the response or a GoogleAPICallError.
        
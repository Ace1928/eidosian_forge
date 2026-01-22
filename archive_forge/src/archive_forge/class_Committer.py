from abc import abstractmethod, ABCMeta
from typing import AsyncContextManager
from google.cloud.pubsublite_v1 import Cursor
class Committer(AsyncContextManager, metaclass=ABCMeta):
    """
    A Committer is able to commit subscribers' completed offsets.
    """

    @abstractmethod
    def commit(self, cursor: Cursor) -> None:
        """
        Start the commit for a cursor.

        Raises:
          GoogleAPICallError: When the committer terminates in failure.
        """
        pass

    @abstractmethod
    async def wait_until_empty(self):
        """
        Flushes pending commits and waits for all outstanding commit responses from the server.

        Raises:
          GoogleAPICallError: When the committer terminates in failure.
        """
        pass
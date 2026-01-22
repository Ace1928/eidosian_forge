from abc import abstractmethod, ABCMeta
from typing import AsyncContextManager

        Discard all outstanding acks and wait for the commit offset to be acknowledged by the server.

        Raises:
          GoogleAPICallError: If the committer has shut down due to a permanent error.
        
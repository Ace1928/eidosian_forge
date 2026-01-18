from abc import ABC, abstractmethod
from typing import (
from langchain_core.runnables import run_in_executor
Get an iterator over keys that match the given prefix.

        Args:
            prefix (str): The prefix to match.

        Returns:
            Iterator[K | str]: An iterator over keys that match the given prefix.

            This method is allowed to return an iterator over either K or str
            depending on what makes more sense for the given store.
        
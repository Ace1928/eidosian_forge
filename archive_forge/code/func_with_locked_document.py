from __future__ import annotations
import logging # isort:skip
from abc import ABCMeta, abstractmethod
from typing import (
from ..core.types import ID
from ..document import Document
from ..settings import settings
@abstractmethod
def with_locked_document(self, func: Callable[[Document], Awaitable[None]]) -> Awaitable[None]:
    """ Runs a function with the document lock held, passing the
        document to the function.

        *Subclasses must implement this method.*

        Args:
            func (callable): function that takes a single parameter (the Document)
                and returns ``None`` or a ``Future``

        Returns:
            a ``Future`` containing the result of the function

        """
    pass
from __future__ import annotations
import logging # isort:skip
import asyncio
from functools import wraps
from typing import (
class UnlockedDocumentProxy:
    """ Wrap a Document object so that only methods that can safely be used
    from unlocked callbacks or threads are exposed. Attempts to otherwise
    access or change the Document results in an exception.

    """

    def __init__(self, doc: Document) -> None:
        """

        """
        self._doc = doc

    def __getattr__(self, attr: str) -> Any:
        """

        """
        raise AttributeError(UNSAFE_DOC_ATTR_USAGE_MSG)

    def add_next_tick_callback(self, callback: Callback) -> NextTickCallback:
        """ Add a "next tick" callback.

        Args:
            callback (callable) :

        """
        return self._doc.add_next_tick_callback(callback)

    def remove_next_tick_callback(self, callback: NextTickCallback) -> None:
        """ Remove a "next tick" callback.

        Args:
            callback (callable) :

        """
        self._doc.remove_next_tick_callback(callback)
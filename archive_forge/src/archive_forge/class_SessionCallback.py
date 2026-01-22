from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Callable, Sequence
from ..core.types import ID
from ..util.tornado import _CallbackGroup
class SessionCallback:
    """ A base class for callback objects associated with Bokeh Documents
    and Sessions.

    """
    _id: ID

    def __init__(self, callback: Callback, *, callback_id: ID) -> None:
        """

         Args:
            callback (callable) :

            id (ID) :

        """
        self._id = callback_id
        self._callback: Callback = callback

    @property
    def id(self) -> ID:
        """ A unique ID for this callback

        """
        return self._id

    @property
    def callback(self) -> Callback:
        """ The callable that this callback wraps.

        """
        return self._callback
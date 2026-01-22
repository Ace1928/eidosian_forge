from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Callable, Sequence
from ..core.types import ID
from ..util.tornado import _CallbackGroup
class PeriodicCallback(SessionCallback):
    """ Represent a callback to execute periodically on the ``IOLoop`` at a
    specified periodic time interval.

    """
    _period: int

    def __init__(self, callback: Callback, period: int, *, callback_id: ID) -> None:
        """

        Args:
            callback (callable) :

            period (int) :

            id (ID) :

        """
        super().__init__(callback=callback, callback_id=callback_id)
        self._period = period

    @property
    def period(self) -> int:
        """ The period time (in milliseconds) that this callback should
        repeat execution at.

        """
        return self._period
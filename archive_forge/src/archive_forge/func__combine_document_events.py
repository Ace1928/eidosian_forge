from __future__ import annotations
import logging # isort:skip
import weakref
from collections import defaultdict
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable
from ..core.enums import HoldPolicy, HoldPolicyType
from ..events import (
from ..model import Model
from ..models.callbacks import Callback as JSEventCallback
from ..util.callback_manager import _check_callback
from .events import (
from .locking import UnlockedDocumentProxy
def _combine_document_events(new_event: DocumentChangedEvent, old_events: list[DocumentChangedEvent]) -> None:
    """ Attempt to combine a new event with a list of previous events.

    The ``old_event`` will be scanned in reverse, and ``.combine(new_event)``
    will be called on each. If a combination can be made, the function
    will return immediately. Otherwise, ``new_event`` will be appended to
    ``old_events``.

    Args:
        new_event (DocumentChangedEvent) :
            The new event to attempt to combine

        old_events (list[DocumentChangedEvent])
            A list of previous events to attempt to combine new_event with

            **This is an "out" parameter**. The values it contains will be
            modified in-place.

    Returns:
        None

    """
    for event in reversed(old_events):
        if event.combine(new_event):
            return
    old_events.append(new_event)
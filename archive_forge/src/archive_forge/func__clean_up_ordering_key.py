import collections
import logging
import typing
from typing import Any, Callable, Iterable, Optional
def _clean_up_ordering_key(self, ordering_key: str) -> None:
    """Clean up state for an ordering key with no pending messages.

        Args
            ordering_key: The ordering key to clean up.
        """
    message_queue = self._pending_ordered_messages.get(ordering_key)
    if message_queue is None:
        _LOGGER.warning('Tried to clean up ordering key that does not exist: %s', ordering_key)
        return
    if len(message_queue) > 0:
        _LOGGER.warning('Tried to clean up ordering key: %s with %d messages remaining.', ordering_key, len(message_queue))
        return
    del self._pending_ordered_messages[ordering_key]
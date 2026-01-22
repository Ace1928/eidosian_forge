import collections
import logging
import typing
from typing import Any, Callable, Iterable, Optional
def _get_next_for_ordering_key(self, ordering_key: str) -> Optional['subscriber.message.Message']:
    """Get next message for ordering key.

        The client should call clean_up_ordering_key() if this method returns
        None.

        Args:
            ordering_key: Ordering key for which to get the next message.

        Returns:
            The next message for this ordering key or None if there aren't any.
        """
    queue_for_key = self._pending_ordered_messages.get(ordering_key)
    if queue_for_key:
        self._size = self._size - 1
        return queue_for_key.popleft()
    return None
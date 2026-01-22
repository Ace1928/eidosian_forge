import abc
import re
import typing
class _retry_always(retry_base):
    """Retry strategy that always rejects any result."""

    def __call__(self, retry_state: 'RetryCallState') -> bool:
        return True
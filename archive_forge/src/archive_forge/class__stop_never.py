import abc
import typing
from pip._vendor.tenacity import _utils
class _stop_never(stop_base):
    """Never stop."""

    def __call__(self, retry_state: 'RetryCallState') -> bool:
        return False
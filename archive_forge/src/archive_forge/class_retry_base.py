import abc
import re
import typing
class retry_base(abc.ABC):
    """Abstract base class for retry strategies."""

    @abc.abstractmethod
    def __call__(self, retry_state: 'RetryCallState') -> bool:
        pass

    def __and__(self, other: 'retry_base') -> 'retry_all':
        return retry_all(self, other)

    def __or__(self, other: 'retry_base') -> 'retry_any':
        return retry_any(self, other)
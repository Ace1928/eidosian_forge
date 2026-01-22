import abc
import re
import typing
class retry_if_not_result(retry_base):
    """Retries if the result refutes a predicate."""

    def __init__(self, predicate: typing.Callable[[typing.Any], bool]) -> None:
        self.predicate = predicate

    def __call__(self, retry_state: 'RetryCallState') -> bool:
        if retry_state.outcome is None:
            raise RuntimeError('__call__() called before outcome was set')
        if not retry_state.outcome.failed:
            return not self.predicate(retry_state.outcome.result())
        else:
            return False
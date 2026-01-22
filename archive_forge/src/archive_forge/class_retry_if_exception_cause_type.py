import abc
import re
import typing
class retry_if_exception_cause_type(retry_base):
    """Retries if any of the causes of the raised exception is of one or more types.

    The check on the type of the cause of the exception is done recursively (until finding
    an exception in the chain that has no `__cause__`)
    """

    def __init__(self, exception_types: typing.Union[typing.Type[BaseException], typing.Tuple[typing.Type[BaseException], ...]]=Exception) -> None:
        self.exception_cause_types = exception_types

    def __call__(self, retry_state: 'RetryCallState') -> bool:
        if retry_state.outcome is None:
            raise RuntimeError('__call__ called before outcome was set')
        if retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            while exc is not None:
                if isinstance(exc.__cause__, self.exception_cause_types):
                    return True
                exc = exc.__cause__
        return False
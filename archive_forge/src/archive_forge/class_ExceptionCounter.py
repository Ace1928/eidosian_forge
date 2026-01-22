from timeit import default_timer
from types import TracebackType
from typing import (
from .decorator import decorate
class ExceptionCounter:

    def __init__(self, counter: 'Counter', exception: Union[Type[BaseException], Tuple[Type[BaseException], ...]]) -> None:
        self._counter = counter
        self._exception = exception

    def __enter__(self) -> None:
        pass

    def __exit__(self, typ: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[TracebackType]) -> Literal[False]:
        if isinstance(value, self._exception):
            self._counter.inc()
        return False

    def __call__(self, f: 'F') -> 'F':

        def wrapped(func, *args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate(f, wrapped)
import bdb
import dataclasses
import os
import sys
from typing import Callable
from typing import cast
from typing import Dict
from typing import final
from typing import Generic
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .reports import BaseReport
from .reports import CollectErrorRepr
from .reports import CollectReport
from .reports import TestReport
from _pytest import timing
from _pytest._code.code import ExceptionChainRepr
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import TerminalRepr
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.nodes import Collector
from _pytest.nodes import Directory
from _pytest.nodes import Item
from _pytest.nodes import Node
from _pytest.outcomes import Exit
from _pytest.outcomes import OutcomeException
from _pytest.outcomes import Skipped
from _pytest.outcomes import TEST_OUTCOME
@final
@dataclasses.dataclass
class CallInfo(Generic[TResult]):
    """Result/Exception info of a function invocation."""
    _result: Optional[TResult]
    excinfo: Optional[ExceptionInfo[BaseException]]
    start: float
    stop: float
    duration: float
    when: Literal['collect', 'setup', 'call', 'teardown']

    def __init__(self, result: Optional[TResult], excinfo: Optional[ExceptionInfo[BaseException]], start: float, stop: float, duration: float, when: Literal['collect', 'setup', 'call', 'teardown'], *, _ispytest: bool=False) -> None:
        check_ispytest(_ispytest)
        self._result = result
        self.excinfo = excinfo
        self.start = start
        self.stop = stop
        self.duration = duration
        self.when = when

    @property
    def result(self) -> TResult:
        """The return value of the call, if it didn't raise.

        Can only be accessed if excinfo is None.
        """
        if self.excinfo is not None:
            raise AttributeError(f'{self!r} has no valid result')
        return cast(TResult, self._result)

    @classmethod
    def from_call(cls, func: Callable[[], TResult], when: Literal['collect', 'setup', 'call', 'teardown'], reraise: Optional[Union[Type[BaseException], Tuple[Type[BaseException], ...]]]=None) -> 'CallInfo[TResult]':
        """Call func, wrapping the result in a CallInfo.

        :param func:
            The function to call. Called without arguments.
        :param when:
            The phase in which the function is called.
        :param reraise:
            Exception or exceptions that shall propagate if raised by the
            function, instead of being wrapped in the CallInfo.
        """
        excinfo = None
        start = timing.time()
        precise_start = timing.perf_counter()
        try:
            result: Optional[TResult] = func()
        except BaseException:
            excinfo = ExceptionInfo.from_current()
            if reraise is not None and isinstance(excinfo.value, reraise):
                raise
            result = None
        precise_stop = timing.perf_counter()
        duration = precise_stop - precise_start
        stop = timing.time()
        return cls(start=start, stop=stop, duration=duration, when=when, result=result, excinfo=excinfo, _ispytest=True)

    def __repr__(self) -> str:
        if self.excinfo is None:
            return f'<CallInfo when={self.when!r} result: {self._result!r}>'
        return f'<CallInfo when={self.when!r} excinfo={self.excinfo!r}>'
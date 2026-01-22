from types import TracebackType
from typing import Callable, List, Optional, Sequence, Type, TypeVar
from unittest import TestCase as PyUnitTestCase
from attrs import Factory, define
from typing_extensions import Literal
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.protocols.amp import AMP, MAX_VALUE_LENGTH
from twisted.python.failure import Failure
from twisted.python.reflect import qual
from twisted.trial._dist import managercommands
from twisted.trial.reporter import TestResult
from ..reporter import TrialFailure
from .stream import chunk, stream
@define
class ReportingResults:
    """
    A mutable container for the result of sending test results back to the
    parent process.

    Since it is possible for these sends to fail asynchronously but the
    L{TestResult} protocol is not well suited for asynchronous result
    reporting, results are collected on an instance of this class and when the
    runner believes the test is otherwise complete, it can collect the results
    and do something with any errors.

    :ivar _reporter: The L{WorkerReporter} this object is associated with.
        This is the object doing the result reporting.

    :ivar _results: A list of L{Deferred} instances representing the results
        of reporting operations.  This is expected to grow over the course of
        the test run and then be inspected by the runner once the test is
        over.  The public interface to this list is via the context manager
        interface.
    """
    _reporter: 'WorkerReporter'
    _results: List[Deferred[object]] = Factory(list)

    def __enter__(self) -> Sequence[Deferred[object]]:
        """
        Begin a new reportable context in which results can be collected.

        :return: A sequence which will contain the L{Deferred} instances
            representing the results of all test result reporting that happens
            while the context manager is active.  The sequence is extended as
            the test runs so its value should not be consumed until the test
            is over.
        """
        return self._results

    def __exit__(self, excType: Type[BaseException], excValue: BaseException, excTraceback: TracebackType) -> Literal[False]:
        """
        End the reportable context.
        """
        self._reporter._reporting = None
        return False

    def record(self, result: Deferred[object]) -> None:
        """
        Record a L{Deferred} instance representing one test result reporting
        operation.
        """
        self._results.append(result)
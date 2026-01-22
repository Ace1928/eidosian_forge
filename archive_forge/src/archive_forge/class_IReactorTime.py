from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorTime(Interface):
    """
    Time methods that a Reactor should implement.
    """

    def seconds() -> float:
        """
        Get the current time in seconds.

        @return: A number-like object of some sort.
        """

    def callLater(delay: float, callable: Callable[..., Any], *args: object, **kwargs: object) -> 'IDelayedCall':
        """
        Call a function later.

        @param delay: the number of seconds to wait.
        @param callable: the callable object to call later.
        @param args: the arguments to call it with.
        @param kwargs: the keyword arguments to call it with.

        @return: An object which provides L{IDelayedCall} and can be used to
                 cancel the scheduled call, by calling its C{cancel()} method.
                 It also may be rescheduled by calling its C{delay()} or
                 C{reset()} methods.
        """

    def getDelayedCalls() -> Sequence['IDelayedCall']:
        """
        See L{twisted.internet.interfaces.IReactorTime.getDelayedCalls}
        """
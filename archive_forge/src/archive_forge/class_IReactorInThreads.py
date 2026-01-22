from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorInThreads(Interface):
    """
    This interface contains the methods exposed by a reactor which will let you
    run functions in another thread.

    @since: 15.4
    """

    def callInThread(callable: Callable[..., Any], *args: object, **kwargs: object) -> None:
        """
        Run the given callable object in a separate thread, with the given
        arguments and keyword arguments.
        """
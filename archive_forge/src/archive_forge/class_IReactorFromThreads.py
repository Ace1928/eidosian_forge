from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorFromThreads(Interface):
    """
    This interface is the set of thread-safe methods which may be invoked on
    the reactor from other threads.

    @since: 15.4
    """

    def callFromThread(callable: Callable[..., Any], *args: object, **kwargs: object) -> None:
        """
        Cause a function to be executed by the reactor thread.

        Use this method when you want to run a function in the reactor's thread
        from another thread.  Calling L{callFromThread} should wake up the main
        thread (where L{reactor.run() <IReactorCore.run>} is executing) and run
        the given callable in that thread.

        If you're writing a multi-threaded application the C{callable}
        may need to be thread safe, but this method doesn't require it as such.
        If you want to call a function in the next mainloop iteration, but
        you're in the same thread, use L{callLater} with a delay of 0.
        """
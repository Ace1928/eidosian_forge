from twisted.internet.interfaces import IReactorThreads, IReactorTime
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.log import msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest

        A
        L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}
        call scheduled from a C{gobject.timeout_add}
        call is run on time.
        
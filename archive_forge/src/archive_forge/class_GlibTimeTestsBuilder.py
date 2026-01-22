from twisted.internet.interfaces import IReactorThreads, IReactorTime
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.log import msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
class GlibTimeTestsBuilder(ReactorBuilder):
    """
    Builder for defining tests relating to L{IReactorTime} for reactors based
    off glib.
    """
    requiredInterfaces = (IReactorTime,)
    _reactors = ['twisted.internet.gireactor.PortableGIReactor' if platform.isWindows() else 'twisted.internet.gireactor.GIReactor']

    def test_timeout_add(self):
        """
        A
        L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}
        call scheduled from a C{gobject.timeout_add}
        call is run on time.
        """
        from gi.repository import GObject
        reactor = self.buildReactor()
        result = []

        def gschedule():
            reactor.callLater(0, callback)
            return 0

        def callback():
            result.append(True)
            reactor.stop()
        reactor.callWhenRunning(GObject.timeout_add, 10, gschedule)
        self.runReactor(reactor, 5)
        self.assertEqual(result, [True])
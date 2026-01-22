from __future__ import annotations
from unittest import skipIf
from twisted.internet.error import ReactorAlreadyRunning
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.trial.unittest import SkipTest, TestCase
class GApplicationRegistrationTests(ReactorBuilder, TestCase):
    """
    GtkApplication and GApplication are supported by
    L{twisted.internet.gtk3reactor} and L{twisted.internet.gireactor}.

    We inherit from L{ReactorBuilder} in order to use some of its
    reactor-running infrastructure, but don't need its test-creation
    functionality.
    """

    def runReactor(self, app: Gio.Application, reactor: gireactor.GIReactor) -> None:
        """
        Register the app, run the reactor, make sure app was activated, and
        that reactor was running, and that reactor can be stopped.
        """
        if not hasattr(app, 'quit'):
            raise SkipTest('Version of PyGObject is too old.')
        result = []

        def stop() -> None:
            result.append('stopped')
            reactor.stop()

        def activate(widget: object) -> None:
            result.append('activated')
            reactor.callLater(0, stop)
        app.connect('activate', activate)
        app.hold()
        reactor.registerGApplication(app)
        ReactorBuilder.runReactor(self, reactor)
        self.assertEqual(result, ['activated', 'stopped'])

    def test_gApplicationActivate(self) -> None:
        """
        L{Gio.Application} instances can be registered with a gireactor.
        """
        self.reactorFactory = lambda: gireactor.GIReactor(useGtk=False)
        reactor = self.buildReactor()
        app = Gio.Application(application_id='com.twistedmatrix.trial.gireactor', flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.runReactor(app, reactor)

    @skipIf(noGtkSkip, noGtkMessage)
    def test_gtkAliases(self) -> None:
        """
        L{twisted.internet.gtk3reactor} is now just a set of compatibility
        aliases for L{twisted.internet.GIReactor}.
        """
        from twisted.internet.gtk3reactor import Gtk3Reactor, PortableGtk3Reactor, install
        self.assertIs(Gtk3Reactor, gireactor.GIReactor)
        self.assertIs(PortableGtk3Reactor, gireactor.PortableGIReactor)
        self.assertIs(install, gireactor.install)
        warnings = self.flushWarnings()
        self.assertEqual(len(warnings), 1)
        self.assertIn('twisted.internet.gtk3reactor was deprecated', warnings[0]['message'])

    @skipIf(noGtkSkip, noGtkMessage)
    def test_gtkApplicationActivate(self) -> None:
        """
        L{Gtk.Application} instances can be registered with a gtk3reactor.
        """
        self.reactorFactory = gireactor.GIReactor
        reactor = self.buildReactor()
        app = Gtk.Application(application_id='com.twistedmatrix.trial.gtk3reactor', flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.runReactor(app, reactor)

    def test_portable(self) -> None:
        """
        L{gireactor.PortableGIReactor} doesn't support application
        registration at this time.
        """
        self.reactorFactory = gireactor.PortableGIReactor
        reactor = self.buildReactor()
        app = Gio.Application(application_id='com.twistedmatrix.trial.gireactor', flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.assertRaises(NotImplementedError, reactor.registerGApplication, app)

    def test_noQuit(self) -> None:
        """
        Older versions of PyGObject lack C{Application.quit}, and so won't
        allow registration.
        """
        self.reactorFactory = lambda: gireactor.GIReactor(useGtk=False)
        reactor = self.buildReactor()
        app = object()
        exc = self.assertRaises(RuntimeError, reactor.registerGApplication, app)
        self.assertTrue(exc.args[0].startswith('Application registration is not'))

    def test_cantRegisterAfterRun(self) -> None:
        """
        It is not possible to register a C{Application} after the reactor has
        already started.
        """
        self.reactorFactory = lambda: gireactor.GIReactor(useGtk=False)
        reactor = self.buildReactor()
        app = Gio.Application(application_id='com.twistedmatrix.trial.gireactor', flags=Gio.ApplicationFlags.FLAGS_NONE)

        def tryRegister() -> None:
            exc = self.assertRaises(ReactorAlreadyRunning, reactor.registerGApplication, app)
            self.assertEqual(exc.args[0], "Can't register application after reactor was started.")
            reactor.stop()
        reactor.callLater(0, tryRegister)
        ReactorBuilder.runReactor(self, reactor)

    def test_cantRegisterTwice(self) -> None:
        """
        It is not possible to register more than one C{Application}.
        """
        self.reactorFactory = lambda: gireactor.GIReactor(useGtk=False)
        reactor = self.buildReactor()
        app = Gio.Application(application_id='com.twistedmatrix.trial.gireactor', flags=Gio.ApplicationFlags.FLAGS_NONE)
        reactor.registerGApplication(app)
        app2 = Gio.Application(application_id='com.twistedmatrix.trial.gireactor2', flags=Gio.ApplicationFlags.FLAGS_NONE)
        exc = self.assertRaises(RuntimeError, reactor.registerGApplication, app2)
        self.assertEqual(exc.args[0], "Can't register more than one application instance.")
from typing import Union
from gi.repository import GLib
from twisted.internet import _glibbase
from twisted.internet.error import ReactorAlreadyRunning
from twisted.python import runtime
class GIReactor(_glibbase.GlibReactorBase):
    """
    GObject-introspection event loop reactor.

    @ivar _gapplication: A C{Gio.Application} instance that was registered
        with C{registerGApplication}.
    """
    _gapplication = None

    def __init__(self, useGtk=False):
        _glibbase.GlibReactorBase.__init__(self, GLib, None)

    def registerGApplication(self, app):
        """
        Register a C{Gio.Application} or C{Gtk.Application}, whose main loop
        will be used instead of the default one.

        We will C{hold} the application so it doesn't exit on its own. In
        versions of C{python-gi} 3.2 and later, we exit the event loop using
        the C{app.quit} method which overrides any holds. Older versions are
        not supported.
        """
        if self._gapplication is not None:
            raise RuntimeError("Can't register more than one application instance.")
        if self._started:
            raise ReactorAlreadyRunning("Can't register application after reactor was started.")
        if not hasattr(app, 'quit'):
            raise RuntimeError('Application registration is not supported in versions of PyGObject prior to 3.2.')
        self._gapplication = app

        def run():
            app.hold()
            app.run(None)
        self._run = run
        self._crash = app.quit
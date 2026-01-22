import signal
from pyglet import app
from pyglet.app.base import PlatformEventLoop, EventLoop
from pyglet.libs.darwin import cocoapy, AutoReleasePool, ObjCSubclass, PyObjectEncoding, ObjCInstance, send_super, \
class _AppDelegate_Implementation:
    _AppDelegate = ObjCSubclass('NSObject', '_AppDelegate')

    @_AppDelegate.method(b'@' + PyObjectEncoding)
    def init(self, pyglet_loop):
        objc = ObjCInstance(send_super(self, 'init'))
        self._pyglet_loop = pyglet_loop
        return objc

    @_AppDelegate.method('v')
    def updatePyglet_(self):
        self._pyglet_loop.nsapp_step()

    @_AppDelegate.method('v@')
    def applicationWillTerminate_(self, notification):
        self._pyglet_loop.is_running = False
        self._pyglet_loop.has_exit = True

    @_AppDelegate.method('v@')
    def applicationDidFinishLaunching_(self, notification):
        self._pyglet_loop._finished_launching = True
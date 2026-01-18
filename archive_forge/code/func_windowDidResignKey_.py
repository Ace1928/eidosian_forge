from pyglet.libs.darwin.cocoapy import ObjCClass, ObjCSubclass, ObjCInstance
from pyglet.libs.darwin.cocoapy import NSApplicationDidHideNotification
from pyglet.libs.darwin.cocoapy import NSApplicationDidUnhideNotification
from pyglet.libs.darwin.cocoapy import send_super, get_selector
from pyglet.libs.darwin.cocoapy import PyObjectEncoding
from pyglet.libs.darwin.cocoapy import quartz
from .systemcursor import SystemCursor
@PygletDelegate.method('v@')
def windowDidResignKey_(self, notification):
    if self._window._mouse_exclusive:
        self._window.set_exclusive_mouse(False)
        self.did_pause_exclusive_mouse = True
        self._window._nswindow.setMovable_(False)
    self._window.set_mouse_platform_visible(True)
    self._window.dispatch_event('on_deactivate')
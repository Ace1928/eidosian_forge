from pyglet.libs.darwin.cocoapy import ObjCClass, ObjCSubclass, ObjCInstance
from pyglet.libs.darwin.cocoapy import NSApplicationDidHideNotification
from pyglet.libs.darwin.cocoapy import NSApplicationDidUnhideNotification
from pyglet.libs.darwin.cocoapy import send_super, get_selector
from pyglet.libs.darwin.cocoapy import PyObjectEncoding
from pyglet.libs.darwin.cocoapy import quartz
from .systemcursor import SystemCursor
@PygletDelegate.method('v@')
def windowDidBecomeKey_(self, notification):
    if self.did_pause_exclusive_mouse:
        self._window.set_exclusive_mouse(True)
        self.did_pause_exclusive_mouse = False
        self._window._nswindow.setMovable_(True)
    self._window.set_mouse_platform_visible()
    self._window.dispatch_event('on_activate')
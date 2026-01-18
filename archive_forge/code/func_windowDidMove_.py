from pyglet.libs.darwin.cocoapy import ObjCClass, ObjCSubclass, ObjCInstance
from pyglet.libs.darwin.cocoapy import NSApplicationDidHideNotification
from pyglet.libs.darwin.cocoapy import NSApplicationDidUnhideNotification
from pyglet.libs.darwin.cocoapy import send_super, get_selector
from pyglet.libs.darwin.cocoapy import PyObjectEncoding
from pyglet.libs.darwin.cocoapy import quartz
from .systemcursor import SystemCursor
@PygletDelegate.method('v@')
def windowDidMove_(self, notification):
    x, y = self._window.get_location()
    self._window.dispatch_event('on_move', x, y)
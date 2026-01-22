from pyglet.input.base import Tablet, TabletCanvas
from pyglet.input.base import TabletCursor, DeviceOpenException
from pyglet.input.linux.x11_xinput import XInputWindowEventDispatcher
from pyglet.input.linux.x11_xinput import get_devices, DeviceResponder
def _proximity_in(self, e):
    cursor = self._cursor_map.get(e.deviceid)
    self.dispatch_event('on_enter', cursor)
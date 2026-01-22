from pyglet.input.base import Tablet, TabletCanvas
from pyglet.input.base import TabletCursor, DeviceOpenException
from pyglet.input.linux.x11_xinput import XInputWindowEventDispatcher
from pyglet.input.linux.x11_xinput import get_devices, DeviceResponder
class XInputTabletCursor(TabletCursor):

    def __init__(self, device):
        super(XInputTabletCursor, self).__init__(device.name)
        self.device = device
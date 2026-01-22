import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
class WintabTabletCanvas(TabletCanvas):
    override_keys = False

    def __init__(self, device, window, msg_base=wintab.WT_DEFBASE):
        super(WintabTabletCanvas, self).__init__(window)
        self.device = device
        self.msg_base = msg_base
        global _extension_masks
        if not _extension_masks:
            _extension_masks = get_extension_masks()
        self.context_info = context_info = wintab.LOGCONTEXT()
        wtinfo(wintab.WTI_DEFSYSCTX, 0, context_info)
        context_info.lcMsgBase = msg_base
        context_info.lcOptions |= wintab.CXO_MESSAGES
        context_info.lcPktData = wintab.PK_CHANGED | wintab.PK_CURSOR | wintab.PK_BUTTONS | wintab.PK_X | wintab.PK_Y | wintab.PK_Z | wintab.PK_NORMAL_PRESSURE | wintab.PK_TANGENT_PRESSURE | wintab.PK_ORIENTATION | _extension_masks
        context_info.lcPktMode = 0
        self._context = lib.WTOpenW(window._hwnd, ctypes.byref(context_info), True)
        if not self._context:
            raise DeviceOpenException("Couldn't open tablet context")
        window._event_handlers[msg_base + wintab.WT_PACKET] = self._event_wt_packet
        window._event_handlers[msg_base + wintab.WT_PROXIMITY] = self._event_wt_proximity
        if _extension_masks:
            window._event_handlers[msg_base + wintab.WT_PACKETEXT] = self._event_wt_packetext
        self._current_cursor = None
        self._pressure_scale = device.pressure_axis.get_scale()
        self._pressure_bias = device.pressure_axis.get_bias()
        self.express_keys = defaultdict(lambda: defaultdict(bool))
        self.express_key_ct = 0
        self.touchrings = []
        self.touchstrips = []
        for tablet_id in range(get_tablet_count()):
            control_count = self.extension_get(wintab.WTX_EXPKEYS2, tablet_id, 0, 0, wintab.TABLET_PROPERTY_CONTROLCOUNT)
            self.express_key_ct = control_count
            assert _debug(f'Controls Found: {control_count}')
            if self.override_keys is True:
                for control_id in range(control_count):
                    function_count = self.extension_get(wintab.WTX_EXPKEYS2, tablet_id, control_id, 0, wintab.TABLET_PROPERTY_FUNCCOUNT)
                    for function_id in range(function_count):
                        self.extension_set(wintab.WTX_EXPKEYS2, tablet_id, control_id, function_id, wintab.TABLET_PROPERTY_OVERRIDE, wintab.BOOL(True))

    def extension_get(self, extension, tablet_id, control_id, function_id, property_id, value_type=wintab.UINT):
        prop = wintab.EXTPROPERTY()
        prop.version = 0
        prop.tabletIndex = tablet_id
        prop.controlIndex = control_id
        prop.functionIndex = function_id
        prop.propertyID = property_id
        prop.reserved = 0
        prop.dataSize = ctypes.sizeof(value_type)
        success = lib.WTExtGet(self._context, extension, ctypes.byref(prop))
        if success:
            return ctypes.cast(prop.data, ctypes.POINTER(value_type)).contents.value
        return 0

    def extension_set(self, extension, tablet_id, control_id, function_id, property_id, value):
        prop = wintab.EXTPROPERTY()
        prop.version = 0
        prop.tabletIndex = tablet_id
        prop.controlIndex = control_id
        prop.functionIndex = function_id
        prop.propertyID = property_id
        prop.reserved = 0
        prop.dataSize = ctypes.sizeof(value)
        prop.data[0] = value.value
        success = lib.WTExtSet(self._context, extension, ctypes.byref(prop))
        if success:
            return True
        return False

    def close(self):
        lib.WTClose(self._context)
        self._context = None
        del self.window._event_handlers[self.msg_base + wintab.WT_PACKET]
        del self.window._event_handlers[self.msg_base + wintab.WT_PROXIMITY]
        if _extension_masks:
            del self.window._event_handlers[self.msg_base + wintab.WT_PACKETEXT]

    def _set_current_cursor(self, cursor_type):
        if self._current_cursor:
            self.dispatch_event('on_leave', self._current_cursor)
        self._current_cursor = self.device._cursor_map.get(cursor_type, None)
        if self._current_cursor:
            self.dispatch_event('on_enter', self._current_cursor)

    @pyglet.window.win32.Win32EventHandler(0)
    def _event_wt_packet(self, msg, wParam, lParam):
        if lParam != self._context:
            return
        packet = wintab.PACKET()
        if lib.WTPacket(self._context, wParam, ctypes.byref(packet)) == 0:
            return
        if not packet.pkChanged:
            return
        window_x, window_y = self.window.get_location()
        window_y = self.window.screen.height - window_y - self.window.height
        x = packet.pkX - window_x
        y = packet.pkY - window_y
        pressure = (packet.pkNormalPressure + self._pressure_bias) * self._pressure_scale
        if self._current_cursor is None:
            self._set_current_cursor(packet.pkCursor)
        self.dispatch_event('on_motion', self._current_cursor, x, y, pressure, 0.0, 0.0, packet.pkButtons)

    @pyglet.window.win32.Win32EventHandler(0)
    def _event_wt_packetext(self, msg, wParam, lParam):
        packet = wintab.PACKETEXT()
        if lib.WTPacket(lParam, wParam, ctypes.byref(packet)) == 0:
            return
        if packet.pkBase.nContext == self._context:
            if packet.pkExpKeys.nControl < self.express_key_ct:
                current_state = self.express_keys[packet.pkExpKeys.nControl][packet.pkExpKeys.nLocation]
                new_state = bool(packet.pkExpKeys.nState)
                if current_state != new_state:
                    event_type = 'on_express_key_press' if new_state else 'on_express_key_release'
                    self.express_keys[packet.pkExpKeys.nControl][packet.pkExpKeys.nLocation] = new_state
                    self.dispatch_event(event_type, packet.pkExpKeys.nControl, packet.pkExpKeys.nLocation)

    @pyglet.window.win32.Win32EventHandler(0)
    def _event_wt_proximity(self, msg, wParam, lParam):
        if wParam != self._context:
            return
        if not lParam & 4294901760:
            return
        if not lParam & 65535:
            self.dispatch_event('on_leave', self._current_cursor)
        self._current_cursor = None

    def on_express_key_press(self, control_id: int, location_id: int):
        """An event called when an ExpressKey is pressed down.

        :Parameters:
            `control_id` : int
                Zero-based index number given to the assigned key by the driver.
                The same control_id may exist in multiple locations, which the location_id is used to differentiate.
            `location_id: int
                Zero-based index indicating side of tablet where control id was found.
                Some tablets may have clusters of ExpressKey's on various areas of the tablet.
                (0 = left, 1 = right, 2 = top, 3 = bottom, 4 = transducer).

        :event:
        """

    def on_express_key_release(self, control_id: int, location_id: int):
        """An event called when an ExpressKey is released.

        :Parameters:
            `control_id` : int
                Zero-based index number given to the assigned key by the driver.
                The same control_id may exist in multiple locations, which the location_id is used to differentiate.
            `location_id: int
                Zero-based index indicating side of tablet where control id was found.
                Some tablets may have clusters of ExpressKey's on various areas of the tablet.
                (0 = left, 1 = right, 2 = top, 3 = bottom, 4 = transducer).

        :event:
        """
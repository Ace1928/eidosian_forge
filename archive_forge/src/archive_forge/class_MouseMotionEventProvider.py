from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
class MouseMotionEventProvider(MotionEventProvider):
    __handlers__ = {}

    def __init__(self, device, args):
        super(MouseMotionEventProvider, self).__init__(device, args)
        self.waiting_event = deque()
        self.touches = {}
        self.counter = 0
        self.current_drag = None
        self.alt_touch = None
        self.disable_on_activity = False
        self.disable_multitouch = False
        self.multitouch_on_demand = False
        self.hover_event = None
        self._disable_hover = False
        self._running = False
        args = args.split(',')
        for arg in args:
            arg = arg.strip()
            if arg == '':
                continue
            elif arg == 'disable_on_activity':
                self.disable_on_activity = True
            elif arg == 'disable_multitouch':
                self.disable_multitouch = True
            elif arg == 'disable_hover':
                self.disable_hover = True
            elif arg == 'multitouch_on_demand':
                self.multitouch_on_demand = True
            else:
                Logger.error('Mouse: unknown parameter <%s>' % arg)

    def _get_disable_hover(self):
        return self._disable_hover

    def _set_disable_hover(self, value):
        if self._disable_hover != value:
            if self._running:
                if value:
                    self._stop_hover_events()
                else:
                    self._start_hover_events()
            self._disable_hover = value
    disable_hover = property(_get_disable_hover, _set_disable_hover)
    'Disables dispatching of hover events if set to ``True``.\n\n    Hover events are enabled by default (`disable_hover` is ``False``). See\n    module documentation if you want to enable/disable hover events through\n    config file.\n\n    .. versionadded:: 2.1.0\n    '

    def start(self):
        """Start the mouse provider"""
        if not EventLoop.window:
            return
        fbind = EventLoop.window.fbind
        fbind('on_mouse_down', self.on_mouse_press)
        fbind('on_mouse_move', self.on_mouse_motion)
        fbind('on_mouse_up', self.on_mouse_release)
        fbind('on_rotate', self.update_touch_graphics)
        fbind('system_size', self.update_touch_graphics)
        if not self.disable_hover:
            self._start_hover_events()
        self._running = True

    def _start_hover_events(self):
        fbind = EventLoop.window.fbind
        fbind('mouse_pos', self.begin_or_update_hover_event)
        fbind('system_size', self.update_hover_event)
        fbind('on_cursor_enter', self.begin_hover_event)
        fbind('on_cursor_leave', self.end_hover_event)
        fbind('on_close', self.end_hover_event)
        fbind('on_rotate', self.update_hover_event)

    def stop(self):
        """Stop the mouse provider"""
        if not EventLoop.window:
            return
        funbind = EventLoop.window.funbind
        funbind('on_mouse_down', self.on_mouse_press)
        funbind('on_mouse_move', self.on_mouse_motion)
        funbind('on_mouse_up', self.on_mouse_release)
        funbind('on_rotate', self.update_touch_graphics)
        funbind('system_size', self.update_touch_graphics)
        if not self.disable_hover:
            self._stop_hover_events()
        self._running = False

    def _stop_hover_events(self):
        funbind = EventLoop.window.funbind
        funbind('mouse_pos', self.begin_or_update_hover_event)
        funbind('system_size', self.update_hover_event)
        funbind('on_cursor_enter', self.begin_hover_event)
        funbind('on_cursor_leave', self.end_hover_event)
        funbind('on_close', self.end_hover_event)
        funbind('on_rotate', self.update_hover_event)

    def test_activity(self):
        if not self.disable_on_activity:
            return False
        for touch in EventLoop.touches:
            if touch.__class__.__name__ == 'KineticMotionEvent':
                continue
            if touch.__class__ != MouseMotionEvent:
                return True
        return False

    def find_touch(self, win, x, y):
        factor = 10.0 / win.system_size[0]
        for touch in self.touches.values():
            if abs(x - touch.sx) < factor and abs(y - touch.sy) < factor:
                return touch
        return None

    def create_event_id(self):
        self.counter += 1
        return self.device + str(self.counter)

    def create_touch(self, win, nx, ny, is_double_tap, do_graphics, button):
        event_id = self.create_event_id()
        args = [nx, ny, button]
        if do_graphics:
            args += [not self.multitouch_on_demand]
        self.current_drag = touch = MouseMotionEvent(self.device, event_id, args, is_touch=True, type_id='touch')
        touch.is_double_tap = is_double_tap
        self.touches[event_id] = touch
        if do_graphics:
            create_flag = not self.disable_multitouch and (not self.multitouch_on_demand)
            touch.update_graphics(win, create_flag)
        self.waiting_event.append(('begin', touch))
        return touch

    def remove_touch(self, win, touch):
        if touch.id in self.touches:
            del self.touches[touch.id]
            touch.update_time_end()
            self.waiting_event.append(('end', touch))
            touch.clear_graphics(win)

    def create_hover(self, win, etype):
        nx, ny = win.to_normalized_pos(*win.mouse_pos)
        nx /= win._density
        ny /= win._density
        args = (nx, ny)
        hover = self.hover_event
        if hover:
            hover.move(args)
        else:
            self.hover_event = hover = MouseMotionEvent(self.device, self.create_event_id(), args, type_id='hover')
        if etype == 'end':
            hover.update_time_end()
            self.hover_event = None
        self.waiting_event.append((etype, hover))

    def on_mouse_motion(self, win, x, y, modifiers):
        nx, ny = win.to_normalized_pos(x, y)
        ny = 1.0 - ny
        if self.current_drag:
            touch = self.current_drag
            touch.move([nx, ny])
            touch.update_graphics(win)
            self.waiting_event.append(('update', touch))
        elif self.alt_touch is not None and 'alt' not in modifiers:
            is_double_tap = 'shift' in modifiers
            self.create_touch(win, nx, ny, is_double_tap, True, [])

    def on_mouse_press(self, win, x, y, button, modifiers):
        if self.test_activity():
            return
        nx, ny = win.to_normalized_pos(x, y)
        ny = 1.0 - ny
        found_touch = self.find_touch(win, nx, ny)
        if found_touch:
            self.current_drag = found_touch
        else:
            is_double_tap = 'shift' in modifiers
            do_graphics = not self.disable_multitouch and (button != 'left' or 'ctrl' in modifiers)
            touch = self.create_touch(win, nx, ny, is_double_tap, do_graphics, button)
            if 'alt' in modifiers:
                self.alt_touch = touch
                self.current_drag = None

    def on_mouse_release(self, win, x, y, button, modifiers):
        if button == 'all':
            for touch in list(self.touches.values()):
                self.remove_touch(win, touch)
            self.current_drag = None
        touch = self.current_drag
        if touch:
            not_right = button in ('left', 'scrollup', 'scrolldown', 'scrollleft', 'scrollright')
            not_ctrl = 'ctrl' not in modifiers
            not_multi = self.disable_multitouch or 'multitouch_sim' not in touch.profile or (not touch.multitouch_sim)
            if not_right and not_ctrl or not_multi:
                self.remove_touch(win, touch)
                self.current_drag = None
            else:
                touch.update_graphics(win, True)
        if self.alt_touch:
            self.remove_touch(win, self.alt_touch)
            self.alt_touch = None

    def update_touch_graphics(self, win, *args):
        for touch in self.touches.values():
            touch.update_graphics(win)

    def begin_or_update_hover_event(self, win, *args):
        etype = 'update' if self.hover_event else 'begin'
        self.create_hover(win, etype)

    def begin_hover_event(self, win, *args):
        if not self.hover_event:
            self.create_hover(win, 'begin')

    def update_hover_event(self, win, *args):
        if self.hover_event:
            self.create_hover(win, 'update')

    def end_hover_event(self, win, *args):
        if self.hover_event:
            self.create_hover(win, 'end')

    def update(self, dispatch_fn):
        """Update the mouse provider (pop event from the queue)"""
        try:
            while True:
                event = self.waiting_event.popleft()
                dispatch_fn(*event)
        except IndexError:
            pass
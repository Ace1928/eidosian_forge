from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.properties import (
class JoyCursor(Widget):
    win = ObjectProperty()
    activated = BooleanProperty(False)
    cursor_width = NumericProperty(1.1)
    cursor_hold = BooleanProperty(False)
    intensity = NumericProperty(4)
    dead_zone = NumericProperty(10000)
    offset_x = NumericProperty(0)
    offset_y = NumericProperty(0)

    def __init__(self, **kwargs):
        super(JoyCursor, self).__init__(**kwargs)
        self.avoid_bring_to_top = False
        self.size_hint = (None, None)
        self.size = (21, 21)
        self.set_cursor()
        with self.canvas:
            Color(rgba=(0.19, 0.64, 0.81, 0.5))
            self.cursor_ox = Line(points=self.cursor_pts[:4], width=self.cursor_width + 0.1)
            self.cursor_oy = Line(points=self.cursor_pts[4:], width=self.cursor_width + 0.1)
            Color(rgba=(1, 1, 1, 0.5))
            self.cursor_x = Line(points=self.cursor_pts[:4], width=self.cursor_width)
            self.cursor_y = Line(points=self.cursor_pts[4:], width=self.cursor_width)
        self.pos = [-i for i in self.size]

    def on_window_children(self, win, *args):
        if self.avoid_bring_to_top or not self.activated:
            return
        self.avoid_bring_to_top = True
        win.remove_widget(self)
        win.add_widget(self)
        self.avoid_bring_to_top = False

    def on_activated(self, instance, activated):
        if activated:
            self.win.add_widget(self)
            self.move = Clock.schedule_interval(self.move_cursor, 0)
            self.win.fbind('on_joy_axis', self.check_cursor)
            self.win.fbind('on_joy_button_down', self.set_intensity)
            self.win.fbind('on_joy_button_down', self.check_dispatch)
            self.win.fbind('mouse_pos', self.stop_cursor)
            mouse_pos = self.win.mouse_pos
            self.pos = (mouse_pos[0] - self.size[0] / 2.0, mouse_pos[1] - self.size[1] / 2.0)
            Logger.info('JoyCursor: joycursor activated')
        else:
            self.pos = [-i for i in self.size]
            Clock.unschedule(self.move)
            self.win.funbind('on_joy_axis', self.check_cursor)
            self.win.funbind('on_joy_button_down', self.set_intensity)
            self.win.funbind('on_joy_button_down', self.check_dispatch)
            self.win.funbind('mouse_pos', self.stop_cursor)
            self.win.remove_widget(self)
            Logger.info('JoyCursor: joycursor deactivated')

    def set_cursor(self, *args):
        px, py = self.pos
        sx, sy = self.size
        self.cursor_pts = [px, py + round(sy / 2.0), px + sx, py + round(sy / 2.0), px + round(sx / 2.0), py, px + round(sx / 2.0), py + sy]

    def check_cursor(self, win, stickid, axisid, value):
        intensity = self.intensity
        dead = self.dead_zone
        if axisid == 3:
            if value < -dead:
                self.offset_x = -intensity
            elif value > dead:
                self.offset_x = intensity
            else:
                self.offset_x = 0
        elif axisid == 4:
            if value < -dead:
                self.offset_y = intensity
            elif value > dead:
                self.offset_y = -intensity
            else:
                self.offset_y = 0
        else:
            self.offset_x = 0
            self.offset_y = 0

    def set_intensity(self, win, stickid, buttonid):
        intensity = self.intensity
        if buttonid == 0 and intensity > 2:
            intensity -= 1
        elif buttonid == 1:
            intensity += 1
        self.intensity = intensity

    def check_dispatch(self, win, stickid, buttonid):
        if buttonid == 6:
            self.cursor_hold = not self.cursor_hold
        if buttonid not in (2, 3, 4, 5, 6):
            return
        x, y = self.center
        y = self.win.system_size[1] - y
        modifiers = []
        actions = {2: 'left', 3: 'right', 4: 'scrollup', 5: 'scrolldown', 6: 'left'}
        button = actions[buttonid]
        self.win.dispatch('on_mouse_down', x, y, button, modifiers)
        if not self.cursor_hold:
            self.win.dispatch('on_mouse_up', x, y, button, modifiers)

    def move_cursor(self, *args):
        self.pos[0] += self.offset_x
        self.pos[1] += self.offset_y
        modifiers = []
        if self.cursor_hold:
            self.win.dispatch('on_mouse_move', self.center[0], self.win.system_size[1] - self.center[1], modifiers)

    def stop_cursor(self, instance, mouse_pos):
        self.offset_x = 0
        self.offset_y = 0
        self.pos = (mouse_pos[0] - self.size[0] / 2.0, mouse_pos[1] - self.size[1] / 2.0)

    def on_pos(self, instance, new_pos):
        self.set_cursor()
        self.cursor_x.points = self.cursor_pts[:4]
        self.cursor_y.points = self.cursor_pts[4:]
        self.cursor_ox.points = self.cursor_pts[:4]
        self.cursor_oy.points = self.cursor_pts[4:]

    def keyboard_shortcuts(self, win, scancode, *args):
        modifiers = args[-1]
        if scancode == 101 and modifiers == ['ctrl']:
            self.activated = not self.activated
            return True
        elif scancode == 27:
            if self.activated:
                self.activated = False
                return True

    def joystick_shortcuts(self, win, stickid, buttonid):
        if buttonid == 7:
            self.activated = not self.activated
            if self.activated:
                self.pos = [round(i / 2.0) for i in win.size]
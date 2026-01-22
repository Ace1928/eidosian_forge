import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
class FigureCanvasGTK4(_FigureCanvasGTK, Gtk.DrawingArea):
    required_interactive_framework = 'gtk4'
    supports_blit = False
    manager_class = _api.classproperty(lambda cls: FigureManagerGTK4)
    _context_is_scaled = False

    def __init__(self, figure=None):
        super().__init__(figure=figure)
        self.set_hexpand(True)
        self.set_vexpand(True)
        self._idle_draw_id = 0
        self._rubberband_rect = None
        self.set_draw_func(self._draw_func)
        self.connect('resize', self.resize_event)
        self.connect('notify::scale-factor', self._update_device_pixel_ratio)
        click = Gtk.GestureClick()
        click.set_button(0)
        click.connect('pressed', self.button_press_event)
        click.connect('released', self.button_release_event)
        self.add_controller(click)
        key = Gtk.EventControllerKey()
        key.connect('key-pressed', self.key_press_event)
        key.connect('key-released', self.key_release_event)
        self.add_controller(key)
        motion = Gtk.EventControllerMotion()
        motion.connect('motion', self.motion_notify_event)
        motion.connect('enter', self.enter_notify_event)
        motion.connect('leave', self.leave_notify_event)
        self.add_controller(motion)
        scroll = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags.VERTICAL)
        scroll.connect('scroll', self.scroll_event)
        self.add_controller(scroll)
        self.set_focusable(True)
        css = Gtk.CssProvider()
        style = '.matplotlib-canvas { background-color: white; }'
        if Gtk.check_version(4, 9, 3) is None:
            css.load_from_data(style, -1)
        else:
            css.load_from_data(style.encode('utf-8'))
        style_ctx = self.get_style_context()
        style_ctx.add_provider(css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        style_ctx.add_class('matplotlib-canvas')

    def destroy(self):
        CloseEvent('close_event', self)._process()

    def set_cursor(self, cursor):
        self.set_cursor_from_name(_backend_gtk.mpl_to_gtk_cursor_name(cursor))

    def _mpl_coords(self, xy=None):
        """
        Convert the *xy* position of a GTK event, or of the current cursor
        position if *xy* is None, to Matplotlib coordinates.

        GTK use logical pixels, but the figure is scaled to physical pixels for
        rendering.  Transform to physical pixels so that all of the down-stream
        transforms work as expected.

        Also, the origin is different and needs to be corrected.
        """
        if xy is None:
            surface = self.get_native().get_surface()
            is_over, x, y, mask = surface.get_device_position(self.get_display().get_default_seat().get_pointer())
        else:
            x, y = xy
        x = x * self.device_pixel_ratio
        y = self.figure.bbox.height - y * self.device_pixel_ratio
        return (x, y)

    def scroll_event(self, controller, dx, dy):
        MouseEvent('scroll_event', self, *self._mpl_coords(), step=dy, modifiers=self._mpl_modifiers(controller))._process()
        return True

    def button_press_event(self, controller, n_press, x, y):
        MouseEvent('button_press_event', self, *self._mpl_coords((x, y)), controller.get_current_button(), modifiers=self._mpl_modifiers(controller))._process()
        self.grab_focus()

    def button_release_event(self, controller, n_press, x, y):
        MouseEvent('button_release_event', self, *self._mpl_coords((x, y)), controller.get_current_button(), modifiers=self._mpl_modifiers(controller))._process()

    def key_press_event(self, controller, keyval, keycode, state):
        KeyEvent('key_press_event', self, self._get_key(keyval, keycode, state), *self._mpl_coords())._process()
        return True

    def key_release_event(self, controller, keyval, keycode, state):
        KeyEvent('key_release_event', self, self._get_key(keyval, keycode, state), *self._mpl_coords())._process()
        return True

    def motion_notify_event(self, controller, x, y):
        MouseEvent('motion_notify_event', self, *self._mpl_coords((x, y)), modifiers=self._mpl_modifiers(controller))._process()

    def enter_notify_event(self, controller, x, y):
        LocationEvent('figure_enter_event', self, *self._mpl_coords((x, y)), modifiers=self._mpl_modifiers())._process()

    def leave_notify_event(self, controller):
        LocationEvent('figure_leave_event', self, *self._mpl_coords(), modifiers=self._mpl_modifiers())._process()

    def resize_event(self, area, width, height):
        self._update_device_pixel_ratio()
        dpi = self.figure.dpi
        winch = width * self.device_pixel_ratio / dpi
        hinch = height * self.device_pixel_ratio / dpi
        self.figure.set_size_inches(winch, hinch, forward=False)
        ResizeEvent('resize_event', self)._process()
        self.draw_idle()

    def _mpl_modifiers(self, controller=None):
        if controller is None:
            surface = self.get_native().get_surface()
            is_over, x, y, event_state = surface.get_device_position(self.get_display().get_default_seat().get_pointer())
        else:
            event_state = controller.get_current_event_state()
        mod_table = [('ctrl', Gdk.ModifierType.CONTROL_MASK), ('alt', Gdk.ModifierType.ALT_MASK), ('shift', Gdk.ModifierType.SHIFT_MASK), ('super', Gdk.ModifierType.SUPER_MASK)]
        return [name for name, mask in mod_table if event_state & mask]

    def _get_key(self, keyval, keycode, state):
        unikey = chr(Gdk.keyval_to_unicode(keyval))
        key = cbook._unikey_or_keysym_to_mplkey(unikey, Gdk.keyval_name(keyval))
        modifiers = [('ctrl', Gdk.ModifierType.CONTROL_MASK, 'control'), ('alt', Gdk.ModifierType.ALT_MASK, 'alt'), ('shift', Gdk.ModifierType.SHIFT_MASK, 'shift'), ('super', Gdk.ModifierType.SUPER_MASK, 'super')]
        mods = [mod for mod, mask, mod_key in modifiers if mod_key != key and state & mask and (not (mod == 'shift' and unikey.isprintable()))]
        return '+'.join([*mods, key])

    def _update_device_pixel_ratio(self, *args, **kwargs):
        if self._set_device_pixel_ratio(self.get_scale_factor()):
            self.draw()

    def _draw_rubberband(self, rect):
        self._rubberband_rect = rect
        self.queue_draw()

    def _draw_func(self, drawing_area, ctx, width, height):
        self.on_draw_event(self, ctx)
        self._post_draw(self, ctx)

    def _post_draw(self, widget, ctx):
        if self._rubberband_rect is None:
            return
        lw = 1
        dash = 3
        if not self._context_is_scaled:
            x0, y0, w, h = (dim / self.device_pixel_ratio for dim in self._rubberband_rect)
        else:
            x0, y0, w, h = self._rubberband_rect
            lw *= self.device_pixel_ratio
            dash *= self.device_pixel_ratio
        x1 = x0 + w
        y1 = y0 + h
        ctx.move_to(x0, y0)
        ctx.line_to(x0, y1)
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y0)
        ctx.move_to(x0, y1)
        ctx.line_to(x1, y1)
        ctx.move_to(x1, y0)
        ctx.line_to(x1, y1)
        ctx.set_antialias(1)
        ctx.set_line_width(lw)
        ctx.set_dash((dash, dash), 0)
        ctx.set_source_rgb(0, 0, 0)
        ctx.stroke_preserve()
        ctx.set_dash((dash, dash), dash)
        ctx.set_source_rgb(1, 1, 1)
        ctx.stroke()

    def on_draw_event(self, widget, ctx):
        pass

    def draw(self):
        if self.is_drawable():
            self.queue_draw()

    def draw_idle(self):
        if self._idle_draw_id != 0:
            return

        def idle_draw(*args):
            try:
                self.draw()
            finally:
                self._idle_draw_id = 0
            return False
        self._idle_draw_id = GLib.idle_add(idle_draw)

    def flush_events(self):
        context = GLib.MainContext.default()
        while context.pending():
            context.iteration(True)
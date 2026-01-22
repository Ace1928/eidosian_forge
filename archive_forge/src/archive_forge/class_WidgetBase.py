import pyglet
from pyglet.event import EventDispatcher
from pyglet.graphics import Group
from pyglet.text.caret import Caret
from pyglet.text.layout import IncrementalTextLayout
class WidgetBase(EventDispatcher):

    def __init__(self, x, y, width, height):
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._bg_group = None
        self._fg_group = None
        self._enabled = True

    def _set_enabled(self, enabled: bool) -> None:
        """Internal hook for setting enabled.

        Override this in subclasses to perform effects when a widget is
        enabled or disabled.
        """
        pass

    @property
    def enabled(self) -> bool:
        """Get/set whether this widget is enabled.

        To react to changes in this value, override
        :py:meth:`._set_enabled` on widgets. For example, you may want
        to cue the user by:

        * Playing an animation and/or sound
        * Setting a highlight color
        * Displaying a toast or notification

        """
        return self._enabled

    @enabled.setter
    def enabled(self, new_enabled: bool) -> None:
        if self._enabled == new_enabled:
            return
        self._enabled = new_enabled
        self._set_enabled(new_enabled)

    def update_groups(self, order):
        pass

    @property
    def x(self):
        """X coordinate of the widget.

        :type: int
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self._update_position()

    @property
    def y(self):
        """Y coordinate of the widget.

        :type: int
        """
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self._update_position()

    @property
    def position(self):
        """The x, y coordinate of the widget as a tuple.

        :type: tuple(int, int)
        """
        return (self._x, self._y)

    @position.setter
    def position(self, values):
        self._x, self._y = values
        self._update_position()

    @property
    def width(self):
        """Width of the widget.

        :type: int
        """
        return self._width

    @property
    def height(self):
        """Height of the widget.

        :type: int
        """
        return self._height

    @property
    def aabb(self):
        """Bounding box of the widget.

        Expressed as (x, y, x + width, y + height)

        :type: (int, int, int, int)
        """
        return (self._x, self._y, self._x + self._width, self._y + self._height)

    @property
    def value(self):
        """Query or set the Widget's value.
        
        This property allows you to set the value of a Widget directly, without any
        user input.  This could be used, for example, to restore Widgets to a
        previous state, or if some event in your program is meant to naturally
        change the same value that the Widget controls.  Note that events are not
        dispatched when changing this property.
        """
        raise NotImplementedError('Value depends on control type!')

    @value.setter
    def value(self, value):
        raise NotImplementedError('Value depends on control type!')

    def _check_hit(self, x, y):
        return self._x < x < self._x + self._width and self._y < y < self._y + self._height

    def _update_position(self):
        raise NotImplementedError('Unable to reposition this Widget')

    def on_key_press(self, symbol, modifiers):
        pass

    def on_key_release(self, symbol, modifiers):
        pass

    def on_mouse_press(self, x, y, buttons, modifiers):
        pass

    def on_mouse_release(self, x, y, buttons, modifiers):
        pass

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        pass

    def on_mouse_motion(self, x, y, dx, dy):
        pass

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        pass

    def on_text(self, text):
        pass

    def on_text_motion(self, motion):
        pass

    def on_text_motion_select(self, motion):
        pass
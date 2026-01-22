from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import CanvasBase, Color, Ellipse, ScissorPush, ScissorPop
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, \
from kivy.uix.relativelayout import RelativeLayout
class TouchRippleBehavior(object):
    """Touch ripple behavior.

    Supposed to be used as mixin on widget classes.

    Ripple behavior does not trigger automatically, concrete implementation
    needs to call :func:`ripple_show` respective :func:`ripple_fade` manually.

    Example
    -------

    Here we create a Label which renders the touch ripple animation on
    interaction::

        class RippleLabel(TouchRippleBehavior, Label):

            def __init__(self, **kwargs):
                super(RippleLabel, self).__init__(**kwargs)

            def on_touch_down(self, touch):
                collide_point = self.collide_point(touch.x, touch.y)
                if collide_point:
                    touch.grab(self)
                    self.ripple_show(touch)
                    return True
                return False

            def on_touch_up(self, touch):
                if touch.grab_current is self:
                    touch.ungrab(self)
                    self.ripple_fade()
                    return True
                return False
    """
    ripple_rad_default = NumericProperty(10)
    'Default radius the animation starts from.\n\n    :attr:`ripple_rad_default` is a :class:`~kivy.properties.NumericProperty`\n    and defaults to `10`.\n    '
    ripple_duration_in = NumericProperty(0.5)
    'Animation duration taken to show the overlay.\n\n    :attr:`ripple_duration_in` is a :class:`~kivy.properties.NumericProperty`\n    and defaults to `0.5`.\n    '
    ripple_duration_out = NumericProperty(0.2)
    'Animation duration taken to fade the overlay.\n\n    :attr:`ripple_duration_out` is a :class:`~kivy.properties.NumericProperty`\n    and defaults to `0.2`.\n    '
    ripple_fade_from_alpha = NumericProperty(0.5)
    'Alpha channel for ripple color the animation starts with.\n\n    :attr:`ripple_fade_from_alpha` is a\n    :class:`~kivy.properties.NumericProperty` and defaults to `0.5`.\n    '
    ripple_fade_to_alpha = NumericProperty(0.8)
    'Alpha channel for ripple color the animation targets to.\n\n    :attr:`ripple_fade_to_alpha` is a :class:`~kivy.properties.NumericProperty`\n    and defaults to `0.8`.\n    '
    ripple_scale = NumericProperty(2.0)
    'Max scale of the animation overlay calculated from max(width/height) of\n    the decorated widget.\n\n    :attr:`ripple_scale` is a :class:`~kivy.properties.NumericProperty`\n    and defaults to `2.0`.\n    '
    ripple_func_in = StringProperty('in_cubic')
    'Animation callback for showing the overlay.\n\n    :attr:`ripple_func_in` is a :class:`~kivy.properties.StringProperty`\n    and defaults to `in_cubic`.\n    '
    ripple_func_out = StringProperty('out_quad')
    'Animation callback for hiding the overlay.\n\n    :attr:`ripple_func_out` is a :class:`~kivy.properties.StringProperty`\n    and defaults to `out_quad`.\n    '
    ripple_rad = NumericProperty(10)
    ripple_pos = ListProperty([0, 0])
    ripple_color = ListProperty((1.0, 1.0, 1.0, 0.5))

    def __init__(self, **kwargs):
        super(TouchRippleBehavior, self).__init__(**kwargs)
        self.ripple_pane = CanvasBase()
        self.canvas.add(self.ripple_pane)
        self.bind(ripple_color=self._ripple_set_color, ripple_pos=self._ripple_set_ellipse, ripple_rad=self._ripple_set_ellipse)
        self.ripple_ellipse = None
        self.ripple_col_instruction = None

    def ripple_show(self, touch):
        """Begin ripple animation on current widget.

        Expects touch event as argument.
        """
        Animation.cancel_all(self, 'ripple_rad', 'ripple_color')
        self._ripple_reset_pane()
        x, y = self.to_window(*self.pos)
        width, height = self.size
        if isinstance(self, RelativeLayout):
            self.ripple_pos = ripple_pos = (touch.x - x, touch.y - y)
        else:
            self.ripple_pos = ripple_pos = (touch.x, touch.y)
        rc = self.ripple_color
        ripple_rad = self.ripple_rad
        self.ripple_color = [rc[0], rc[1], rc[2], self.ripple_fade_from_alpha]
        with self.ripple_pane:
            ScissorPush(x=int(round(x)), y=int(round(y)), width=int(round(width)), height=int(round(height)))
            self.ripple_col_instruction = Color(rgba=self.ripple_color)
            self.ripple_ellipse = Ellipse(size=(ripple_rad, ripple_rad), pos=(ripple_pos[0] - ripple_rad / 2.0, ripple_pos[1] - ripple_rad / 2.0))
            ScissorPop()
        anim = Animation(ripple_rad=max(width, height) * self.ripple_scale, t=self.ripple_func_in, ripple_color=[rc[0], rc[1], rc[2], self.ripple_fade_to_alpha], duration=self.ripple_duration_in)
        anim.start(self)

    def ripple_fade(self):
        """Finish ripple animation on current widget.
        """
        Animation.cancel_all(self, 'ripple_rad', 'ripple_color')
        width, height = self.size
        rc = self.ripple_color
        duration = self.ripple_duration_out
        anim = Animation(ripple_rad=max(width, height) * self.ripple_scale, ripple_color=[rc[0], rc[1], rc[2], 0.0], t=self.ripple_func_out, duration=duration)
        anim.bind(on_complete=self._ripple_anim_complete)
        anim.start(self)

    def _ripple_set_ellipse(self, instance, value):
        ellipse = self.ripple_ellipse
        if not ellipse:
            return
        ripple_pos = self.ripple_pos
        ripple_rad = self.ripple_rad
        ellipse.size = (ripple_rad, ripple_rad)
        ellipse.pos = (ripple_pos[0] - ripple_rad / 2.0, ripple_pos[1] - ripple_rad / 2.0)

    def _ripple_set_color(self, instance, value):
        if not self.ripple_col_instruction:
            return
        self.ripple_col_instruction.rgba = value

    def _ripple_anim_complete(self, anim, instance):
        self._ripple_reset_pane()

    def _ripple_reset_pane(self):
        self.ripple_rad = self.ripple_rad_default
        self.ripple_pane.clear()
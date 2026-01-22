from kivy.animation import Animation
from kivy.properties import (
from kivy.uix.anchorlayout import AnchorLayout
class ModalView(AnchorLayout):
    """ModalView class. See module documentation for more information.

    :Events:
        `on_pre_open`:
            Fired before the ModalView is opened. When this event is fired
            ModalView is not yet added to window.
        `on_open`:
            Fired when the ModalView is opened.
        `on_pre_dismiss`:
            Fired before the ModalView is closed.
        `on_dismiss`:
            Fired when the ModalView is closed. If the callback returns True,
            the dismiss will be canceled.

    .. versionchanged:: 1.11.0
        Added events `on_pre_open` and `on_pre_dismiss`.

    .. versionchanged:: 2.0.0
        Added property 'overlay_color'.

    .. versionchanged:: 2.1.0
        Marked `attach_to` property as deprecated.

    """
    auto_dismiss = BooleanProperty(True)
    'This property determines if the view is automatically\n    dismissed when the user clicks outside it.\n\n    :attr:`auto_dismiss` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to True.\n    '
    attach_to = ObjectProperty(None, deprecated=True)
    'If a widget is set on attach_to, the view will attach to the nearest\n    parent window of the widget. If none is found, it will attach to the\n    main/global Window.\n\n    :attr:`attach_to` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None.\n    '
    background_color = ColorProperty([1, 1, 1, 1])
    "Background color, in the format (r, g, b, a).\n\n    This acts as a *multiplier* to the texture color. The default\n    texture is grey, so just setting the background color will give\n    a darker result. To set a plain color, set the\n    :attr:`background_normal` to ``''``.\n\n    The :attr:`background_color` is a\n    :class:`~kivy.properties.ColorProperty` and defaults to [1, 1, 1, 1].\n\n    .. versionchanged:: 2.0.0\n        Changed behavior to affect the background of the widget itself, not\n        the overlay dimming.\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    "
    background = StringProperty('atlas://data/images/defaulttheme/modalview-background')
    "Background image of the view used for the view background.\n\n    :attr:`background` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'atlas://data/images/defaulttheme/modalview-background'.\n    "
    border = ListProperty([16, 16, 16, 16])
    'Border used for :class:`~kivy.graphics.vertex_instructions.BorderImage`\n    graphics instruction. Used for the :attr:`background_normal` and the\n    :attr:`background_down` properties. Can be used when using custom\n    backgrounds.\n\n    It must be a list of four values: (bottom, right, top, left). Read the\n    BorderImage instructions for more information about how to use it.\n\n    :attr:`border` is a :class:`~kivy.properties.ListProperty` and defaults to\n    (16, 16, 16, 16).\n    '
    overlay_color = ColorProperty([0, 0, 0, 0.7])
    'Overlay color in the format (r, g, b, a).\n    Used for dimming the window behind the modal view.\n\n    :attr:`overlay_color` is a :class:`~kivy.properties.ColorProperty` and\n    defaults to [0, 0, 0, .7].\n\n    .. versionadded:: 2.0.0\n    '
    _anim_alpha = NumericProperty(0)
    _anim_duration = NumericProperty(0.1)
    _window = ObjectProperty(allownone=True, rebind=True)
    _is_open = BooleanProperty(False)
    _touch_started_inside = None
    __events__ = ('on_pre_open', 'on_open', 'on_pre_dismiss', 'on_dismiss')

    def __init__(self, **kwargs):
        self._parent = None
        super(ModalView, self).__init__(**kwargs)

    def open(self, *_args, **kwargs):
        """Display the modal in the Window.

        When the view is opened, it will be faded in with an animation. If you
        don't want the animation, use::

            view.open(animation=False)

        """
        from kivy.core.window import Window
        if self._is_open:
            return
        self._window = Window
        self._is_open = True
        self.dispatch('on_pre_open')
        Window.add_widget(self)
        Window.bind(on_resize=self._align_center, on_keyboard=self._handle_keyboard)
        self.center = Window.center
        self.fbind('center', self._align_center)
        self.fbind('size', self._align_center)
        if kwargs.get('animation', True):
            ani = Animation(_anim_alpha=1.0, d=self._anim_duration)
            ani.bind(on_complete=lambda *_args: self.dispatch('on_open'))
            ani.start(self)
        else:
            self._anim_alpha = 1.0
            self.dispatch('on_open')

    def dismiss(self, *_args, **kwargs):
        """ Close the view if it is open.

        If you really want to close the view, whatever the on_dismiss
        event returns, you can use the *force* keyword argument::

            view = ModalView()
            view.dismiss(force=True)

        When the view is dismissed, it will be faded out before being
        removed from the parent. If you don't want this animation, use::

            view.dismiss(animation=False)

        """
        if not self._is_open:
            return
        self.dispatch('on_pre_dismiss')
        if self.dispatch('on_dismiss') is True:
            if kwargs.get('force', False) is not True:
                return
        if kwargs.get('animation', True):
            Animation(_anim_alpha=0.0, d=self._anim_duration).start(self)
        else:
            self._anim_alpha = 0
            self._real_remove_widget()

    def _align_center(self, *_args):
        if self._is_open:
            self.center = self._window.center

    def on_motion(self, etype, me):
        super().on_motion(etype, me)
        return True

    def on_touch_down(self, touch):
        """ touch down event handler. """
        self._touch_started_inside = self.collide_point(*touch.pos)
        if not self.auto_dismiss or self._touch_started_inside:
            super().on_touch_down(touch)
        return True

    def on_touch_move(self, touch):
        """ touch moved event handler. """
        if not self.auto_dismiss or self._touch_started_inside:
            super().on_touch_move(touch)
        return True

    def on_touch_up(self, touch):
        """ touch up event handler. """
        if self.auto_dismiss and self._touch_started_inside is False:
            self.dismiss()
        else:
            super().on_touch_up(touch)
        self._touch_started_inside = None
        return True

    def on__anim_alpha(self, _instance, value):
        """ animation progress callback. """
        if value == 0 and self._is_open:
            self._real_remove_widget()

    def _real_remove_widget(self):
        if not self._is_open:
            return
        self._window.remove_widget(self)
        self._window.unbind(on_resize=self._align_center, on_keyboard=self._handle_keyboard)
        self._is_open = False
        self._window = None

    def on_pre_open(self):
        """ default pre-open event handler. """

    def on_open(self):
        """ default open event handler. """

    def on_pre_dismiss(self):
        """ default pre-dismiss event handler. """

    def on_dismiss(self):
        """ default dismiss event handler. """

    def _handle_keyboard(self, _window, key, *_args):
        if key == 27 and self.auto_dismiss:
            self.dismiss()
            return True
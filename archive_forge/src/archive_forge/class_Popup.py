from kivy.core.text import DEFAULT_FONT
from kivy.uix.modalview import ModalView
from kivy.properties import (StringProperty, ObjectProperty, OptionProperty,
class Popup(ModalView):
    """Popup class. See module documentation for more information.

    :Events:
        `on_open`:
            Fired when the Popup is opened.
        `on_dismiss`:
            Fired when the Popup is closed. If the callback returns True, the
            dismiss will be canceled.
    """
    title = StringProperty('No title')
    "String that represents the title of the popup.\n\n    :attr:`title` is a :class:`~kivy.properties.StringProperty` and defaults to\n    'No title'.\n    "
    title_size = NumericProperty('14sp')
    "Represents the font size of the popup title.\n\n    .. versionadded:: 1.6.0\n\n    :attr:`title_size` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to '14sp'.\n    "
    title_align = OptionProperty('left', options=['left', 'center', 'right', 'justify'])
    "Horizontal alignment of the title.\n\n    .. versionadded:: 1.9.0\n\n    :attr:`title_align` is a :class:`~kivy.properties.OptionProperty` and\n    defaults to 'left'. Available options are left, center, right and justify.\n    "
    title_font = StringProperty(DEFAULT_FONT)
    "Font used to render the title text.\n\n    .. versionadded:: 1.9.0\n\n    :attr:`title_font` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'Roboto'. This value is taken\n    from :class:`~kivy.config.Config`.\n    "
    content = ObjectProperty(None)
    'Content of the popup that is displayed just under the title.\n\n    :attr:`content` is an :class:`~kivy.properties.ObjectProperty` and defaults\n    to None.\n    '
    title_color = ColorProperty([1, 1, 1, 1])
    'Color used by the Title.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`title_color` is a :class:`~kivy.properties.ColorProperty` and\n    defaults to [1, 1, 1, 1].\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    '
    separator_color = ColorProperty([47 / 255.0, 167 / 255.0, 212 / 255.0, 1.0])
    'Color used by the separator between title and content.\n\n    .. versionadded:: 1.1.0\n\n    :attr:`separator_color` is a :class:`~kivy.properties.ColorProperty` and\n    defaults to [47 / 255., 167 / 255., 212 / 255., 1.].\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    '
    separator_height = NumericProperty('2dp')
    'Height of the separator.\n\n    .. versionadded:: 1.1.0\n\n    :attr:`separator_height` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 2dp.\n    '
    _container = ObjectProperty(None)

    def add_widget(self, widget, *args, **kwargs):
        if self._container:
            if self.content:
                raise PopupException('Popup can have only one widget as content')
            self.content = widget
        else:
            super(Popup, self).add_widget(widget, *args, **kwargs)

    def on_content(self, instance, value):
        if self._container:
            self._container.clear_widgets()
            self._container.add_widget(value)

    def on__container(self, instance, value):
        if value is None or self.content is None:
            return
        self._container.clear_widgets()
        self._container.add_widget(self.content)

    def on_touch_down(self, touch):
        if self.disabled and self.collide_point(*touch.pos):
            return True
        return super(Popup, self).on_touch_down(touch)
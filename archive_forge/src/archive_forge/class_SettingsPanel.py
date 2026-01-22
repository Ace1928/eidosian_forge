import json
import os
import kivy.utils as utils
from kivy.factory import Factory
from kivy.metrics import dp
from kivy.config import ConfigParser
from kivy.animation import Animation
from kivy.compat import string_types, text_type
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.tabbedpanel import TabbedPanelHeader
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.colorpicker import ColorPicker
from kivy.uix.scrollview import ScrollView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, StringProperty, ListProperty, \
class SettingsPanel(GridLayout):
    """This class is used to construct panel settings, for use with a
    :class:`Settings` instance or subclass.
    """
    title = StringProperty('Default title')
    'Title of the panel. The title will be reused by the :class:`Settings` in\n    the sidebar.\n    '
    config = ObjectProperty(None, allownone=True)
    'A :class:`kivy.config.ConfigParser` instance. See module documentation\n    for more information.\n    '
    settings = ObjectProperty(None)
    'A :class:`Settings` instance that will be used to fire the\n    `on_config_change` event.\n    '

    def __init__(self, **kwargs):
        kwargs.setdefault('cols', 1)
        super(SettingsPanel, self).__init__(**kwargs)

    def on_config(self, instance, value):
        if value is None:
            return
        if not isinstance(value, ConfigParser):
            raise Exception('Invalid config object, you must use akivy.config.ConfigParser, not another one !')

    def get_value(self, section, key):
        """Return the value of the section/key from the :attr:`config`
        ConfigParser instance. This function is used by :class:`SettingItem` to
        get the value for a given section/key.

        If you don't want to use a ConfigParser instance, you might want to
        override this function.
        """
        config = self.config
        if not config:
            return
        return config.get(section, key)

    def set_value(self, section, key, value):
        current = self.get_value(section, key)
        if current == value:
            return
        config = self.config
        if config:
            config.set(section, key, value)
            config.write()
        settings = self.settings
        if settings:
            settings.dispatch('on_config_change', config, section, key, value)
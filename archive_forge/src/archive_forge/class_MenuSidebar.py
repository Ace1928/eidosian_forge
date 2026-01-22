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
class MenuSidebar(FloatLayout):
    """The menu used by :class:`InterfaceWithSidebar`. It provides a
    sidebar with an entry for each settings panel, which the user may
    click to select.

    """
    selected_uid = NumericProperty(0)
    'The uid of the currently selected panel. This may be used to switch\n    between displayed panels, e.g. by binding it to the\n    :attr:`~ContentPanel.current_uid` of a :class:`ContentPanel`.\n\n    :attr:`selected_uid` is a\n    :class:`~kivy.properties.NumericProperty` and defaults to 0.\n\n    '
    buttons_layout = ObjectProperty(None)
    '(internal) Reference to the GridLayout that contains individual\n    settings panel menu buttons.\n\n    :attr:`buttons_layout` is an\n    :class:`~kivy.properties.ObjectProperty` and defaults to None.\n\n    '
    close_button = ObjectProperty(None)
    "(internal) Reference to the widget's Close button.\n\n    :attr:`buttons_layout` is an\n    :class:`~kivy.properties.ObjectProperty` and defaults to None.\n\n    "

    def add_item(self, name, uid):
        """This method is used to add new panels to the menu.

        :Parameters:
            `name`:
                The name (a string) of the panel. It should be used
                to represent the panel in the menu.
            `uid`:
                The name (an int) of the panel. It should be used internally
                to represent the panel and used to set self.selected_uid when
                the panel is changed.

        """
        label = SettingSidebarLabel(text=name, uid=uid, menu=self)
        if len(self.buttons_layout.children) == 0:
            label.selected = True
        if self.buttons_layout is not None:
            self.buttons_layout.add_widget(label)

    def on_selected_uid(self, *args):
        """(internal) unselects any currently selected menu buttons, unless
        they represent the current panel.

        """
        for button in self.buttons_layout.children:
            if button.uid != self.selected_uid:
                button.selected = False
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
def _create_popup(self, instance):
    content = BoxLayout(orientation='vertical', spacing='5dp')
    popup_width = min(0.95 * Window.width, dp(500))
    self.popup = popup = Popup(content=content, title=self.title, size_hint=(None, None), size=(popup_width, '400dp'))
    popup.height = len(self.options) * dp(55) + dp(150)
    content.add_widget(Widget(size_hint_y=None, height=1))
    uid = str(self.uid)
    for option in self.options:
        state = 'down' if option == self.value else 'normal'
        btn = ToggleButton(text=option, state=state, group=uid)
        btn.bind(on_release=self._set_option)
        content.add_widget(btn)
    content.add_widget(SettingSpacer())
    btn = Button(text='Cancel', size_hint_y=None, height=dp(50))
    btn.bind(on_release=popup.dismiss)
    content.add_widget(btn)
    popup.open()
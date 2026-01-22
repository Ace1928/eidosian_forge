import kivy
import weakref
from functools import partial
from itertools import chain
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.treeview import TreeViewNode, TreeView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.modalview import ModalView
from kivy.graphics import Color, Rectangle, PushMatrix, PopMatrix
from kivy.graphics.context_instructions import Transform
from kivy.graphics.transformation import Matrix
from kivy.properties import (ObjectProperty, BooleanProperty, ListProperty,
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.lang import Builder
class ConsoleAddonSelect(ConsoleAddon):

    def init(self):
        self.btn = ConsoleToggleButton(text=u'Select')
        self.btn.bind(state=self.on_button_state)
        self.console.add_toolbar_widget(self.btn)
        self.console.bind(inspect_enabled=self.on_inspect_enabled)

    def on_inspect_enabled(self, instance, value):
        self.btn.state = 'down' if value else 'normal'

    def on_button_state(self, instance, value):
        self.console.inspect_enabled = value == 'down'
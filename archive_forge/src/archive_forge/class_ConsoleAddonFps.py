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
class ConsoleAddonFps(ConsoleAddon):
    _update_ev = None

    def init(self):
        self.lbl = ConsoleLabel(text='0 Fps')
        self.console.add_toolbar_widget(self.lbl, right=True)

    def activate(self):
        ev = self._update_ev
        if ev is None:
            self._update_ev = Clock.schedule_interval(self.update_fps, 1 / 2.0)
        else:
            ev()

    def deactivated(self):
        if self._update_ev is not None:
            self._update_ev.cancel()

    def update_fps(self, *args):
        fps = Clock.get_fps()
        self.lbl.text = '{} Fps'.format(int(fps))
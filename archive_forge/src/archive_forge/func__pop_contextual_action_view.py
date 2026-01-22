from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.checkbox import CheckBox
from kivy.uix.spinner import Spinner
from kivy.uix.label import Label
from kivy.config import Config
from kivy.properties import ObjectProperty, NumericProperty, BooleanProperty, \
from kivy.metrics import sp
from kivy.lang import Builder
from functools import partial
def _pop_contextual_action_view(self):
    """Remove the current ContextualActionView and display either the
           previous one or the ActionView.
        """
    self._stack_cont_action_view.pop()
    self.clear_widgets()
    if self._stack_cont_action_view == []:
        super(ActionBar, self).add_widget(self.action_view)
    else:
        super(ActionBar, self).add_widget(self._stack_cont_action_view[-1])
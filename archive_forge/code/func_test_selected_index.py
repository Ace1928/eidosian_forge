from unittest import TestCase
from traitlets import TraitError
from ipywidgets.widgets import Accordion, Tab, Stack, HTML
def test_selected_index(self):
    widget = self.widget(self.children, selected_index=1)
    state = widget.get_state()
    assert state['selected_index'] == 1
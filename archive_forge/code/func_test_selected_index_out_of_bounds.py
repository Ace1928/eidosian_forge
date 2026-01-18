from unittest import TestCase
from traitlets import TraitError
from ipywidgets.widgets import Accordion, Tab, Stack, HTML
def test_selected_index_out_of_bounds(self):
    with self.assertRaises(TraitError):
        self.widget(self.children, selected_index=-1)
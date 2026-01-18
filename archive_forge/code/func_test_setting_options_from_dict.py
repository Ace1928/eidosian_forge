import inspect
from unittest import TestCase
from traitlets import TraitError
from ipywidgets import Dropdown, SelectionSlider, Select
def test_setting_options_from_dict(self):
    d = Dropdown()
    assert d.options == ()
    d.options = {'One': 1, 'Two': 2, 'Three': 3}
    assert d.get_state('_options_labels') == {'_options_labels': ('One', 'Two', 'Three')}
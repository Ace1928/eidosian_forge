import inspect
from unittest import TestCase
from traitlets import TraitError
from ipywidgets import Dropdown, SelectionSlider, Select
def test_setting_options_from_list(self):
    d = Dropdown()
    assert d.options == ()
    d.options = ['One', 'Two', 'Three']
    assert d.get_state('_options_labels') == {'_options_labels': ('One', 'Two', 'Three')}
from unittest import TestCase
from unittest import mock
import pytest
import traitlets
import ipywidgets as widgets
from ipywidgets.widgets.widget_templates import LayoutProperties
def test_pass_layout_options(self):
    """test whether the extra layout options of the template class are
           passed down to Layout object"""
    button1 = widgets.Button()
    button2 = widgets.Button()
    button3 = widgets.Button()
    button4 = widgets.Button()
    box = widgets.TwoByTwoLayout(top_left=button1, top_right=button2, bottom_left=button3, bottom_right=button4, grid_gap='10px', justify_content='center', align_items='center')
    assert box.layout.grid_gap == '10px'
    assert box.layout.justify_content == 'center'
    assert box.layout.align_items == 'center'
    layout = widgets.Layout(grid_gap='10px', justify_content='center', align_items='center')
    box = widgets.TwoByTwoLayout(top_left=button1, top_right=button2, bottom_left=button3, bottom_right=button4, layout=layout)
    assert box.layout.grid_gap == '10px'
    assert box.layout.justify_content == 'center'
    assert box.layout.align_items == 'center'
    layout = widgets.Layout(grid_gap='10px', justify_content='center', align_items='center')
    box = widgets.TwoByTwoLayout(top_left=button1, top_right=button2, bottom_left=button3, bottom_right=button4, layout=layout, grid_gap='30px')
    assert box.layout.grid_gap == '30px'
    assert box.layout.justify_content == 'center'
    assert box.layout.align_items == 'center'
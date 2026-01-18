import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_select_text_option_with_equality():
    opts = {'A': 'ABC', '1': 'DEF'}
    select = Select(options=opts, value='DEF', name='Select')
    assert select.value == 'DEF'
    select.value = 'ABC'
    assert select.value == 'ABC'
    select.value = 'DEF'
    assert select.value == 'DEF'
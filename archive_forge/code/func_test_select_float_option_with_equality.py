import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_select_float_option_with_equality():
    opts = {'A': 3.14, '1': 2.0}
    select = Select(options=opts, value=3.14, name='Select')
    assert select.value == 3.14
    select.value = 2
    assert select.value == 2.0
    select.value = 3.14
    assert select.value == 3.14
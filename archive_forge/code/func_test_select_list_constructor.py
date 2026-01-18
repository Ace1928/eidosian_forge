import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_select_list_constructor():
    select = Select(options=['A', 1], value=1)
    assert select.options == ['A', 1]
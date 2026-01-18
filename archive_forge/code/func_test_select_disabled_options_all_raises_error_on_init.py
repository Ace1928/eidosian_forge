import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
@pytest.mark.parametrize('options', [[10, 20], dict(A=10, B=20)], ids=['list', 'dict'])
@pytest.mark.parametrize('size', [1, 2], ids=['size=1', 'size>1'])
def test_select_disabled_options_all_raises_error_on_init(options, size):
    with pytest.raises(ValueError, match='All the options'):
        Select(options=options, disabled_options=[10, 20], size=size)
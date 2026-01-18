import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_toggle_group_error_init(document, comm):
    with pytest.raises(ValueError):
        ToggleGroup(options={'A': 'A', '1': 1, 'C': object}, value=1, name='RadioButtonGroup', widget_type='button', behavior='check')
    with pytest.raises(ValueError):
        ToggleGroup(options={'A': 'A', '1': 1, 'C': object}, value=[1, object], name='RadioButtonGroup', widget_type='button', behavior='radio')
    with pytest.raises(ValueError):
        ToggleGroup(options={'A': 'A', '1': 1, 'C': object}, value=[1, object], name='RadioButtonGroup', widget_type='buttons')
    with pytest.raises(ValueError):
        ToggleGroup(options={'A': 'A', '1': 1, 'C': object}, value=[1, object], name='RadioButtonGroup', behavior='checks')
import param
import pytest
from panel.io import block_comm
from panel.layout import Row
from panel.links import CallbackGenerator
from panel.tests.util import check_layoutable_properties
from panel.util import param_watchers
from panel.widgets import (
from panel.widgets.tables import BaseTable
@pytest.mark.parametrize('widget', all_widgets)
def test_widget_clone(widget):
    w = widget()
    clone = w.clone()
    assert [(k, v) for k, v in sorted(w.param.values().items()) if k != 'name'] == [(k, v) for k, v in sorted(clone.param.values().items()) if k != 'name']
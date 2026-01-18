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
def test_widget_linkable_params(widget, document, comm):
    w = widget()
    controls = w.controls(jslink=True)
    layout = Row(w, controls)
    try:
        CallbackGenerator.error = True
        layout.get_root(document, comm)
    finally:
        CallbackGenerator.error = False
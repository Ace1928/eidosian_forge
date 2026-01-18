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
def test_widget_signature(widget):
    from inspect import signature
    parameters = signature(widget).parameters
    if getattr(getattr(widget, '_param__private', object), 'signature', None):
        pytest.skip('Signature already set by Param')
    assert len(parameters) == 1
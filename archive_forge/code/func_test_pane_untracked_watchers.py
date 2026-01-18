import param
import pytest
import panel as pn
from panel.chat import ChatMessage
from panel.config import config
from panel.interact import interactive
from panel.io.loading import LOADING_INDICATOR_CSS_CLASS
from panel.layout import Row
from panel.links import CallbackGenerator
from panel.pane import (
from panel.param import (
from panel.tests.util import check_layoutable_properties
from panel.util import param_watchers
@pytest.mark.parametrize('pane', all_panes + [Bokeh])
def test_pane_untracked_watchers(pane, document, comm):
    try:
        p = pane()
    except ImportError:
        pytest.skip('Dependent library could not be imported.')
    watchers = [w for pwatchers in param_watchers(p).values() for awatchers in pwatchers.values() for w in awatchers]
    assert len([wfn for wfn in watchers if wfn not in p._internal_callbacks and (not hasattr(wfn.fn, '_watcher_name'))]) == 0
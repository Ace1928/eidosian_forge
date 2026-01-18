import os
from pathlib import Path
import bokeh
from packaging.version import Version
from panel.config import config, panel_extension as extension
from panel.io.resources import (
from panel.io.state import set_curdoc
from panel.theme.native import Native
from panel.widgets import Button
def test_resolve_custom_path_abs_input_relative_to():
    assert str(resolve_custom_path(Button, PANEL_DIR / 'widgets' / 'button.py', relative=True)) == 'button.py'
import glob
import json
import os
from io import StringIO
import pytest
from bokeh.models import CustomJS
from panel import Row
from panel.config import config
from panel.io.embed import embed_state
from panel.pane import Str
from panel.param import Param
from panel.widgets import (
def test_save_embed(tmpdir):
    checkbox = Checkbox()
    string = Str()
    checkbox.link(string, value='object')
    panel = Row(checkbox, string)
    filename = os.path.join(str(tmpdir), 'test.html')
    panel.save(filename, embed=True)
    assert os.path.isfile(filename)
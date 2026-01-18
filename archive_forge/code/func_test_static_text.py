from datetime import date, datetime
from pathlib import Path
import numpy as np
import pytest
from bokeh.models.widgets import FileInput as BkFileInput
from panel import config
from panel.widgets import (
def test_static_text(document, comm):
    text = StaticText(value='ABC', name='Text:')
    widget = text.get_root(document, comm=comm)
    assert isinstance(widget, text._widget_type)
    assert widget.text == '<b>Text:</b>: ABC'
    text.value = 'CBA'
    assert widget.text == '<b>Text:</b>: CBA'
    text.value = '<b>Text:</b>: ABC'
    assert widget.text == '<b>Text:</b>: ABC'
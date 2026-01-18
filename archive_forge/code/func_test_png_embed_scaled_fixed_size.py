import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
def test_png_embed_scaled_fixed_size(document, comm):
    png = PNG(PNG_FILE, width=400, embed=True)
    model = png.get_root(document, comm)
    assert 'width: 400px' in model.text
    assert 'height: 300px' in model.text
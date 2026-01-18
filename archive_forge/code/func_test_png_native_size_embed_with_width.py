import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
def test_png_native_size_embed_with_width(document, comm):
    png = PNG(PNG_FILE, embed=True, width=200)
    model = png.get_root(document, comm)
    assert 'width: 200px' in model.text
    assert 'height: 150px' in model.text
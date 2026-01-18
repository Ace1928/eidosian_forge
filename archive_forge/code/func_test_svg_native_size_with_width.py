import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
@pytest.mark.parametrize('embed', [False, True])
def test_svg_native_size_with_width(embed, document, comm):
    svg = SVG(SVG_FILE, embed=embed, width=200)
    model = svg.get_root(document, comm)
    assert 'width: 200px' in model.text
    assert 'height: auto' in model.text
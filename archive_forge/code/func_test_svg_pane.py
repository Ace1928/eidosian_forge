import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
def test_svg_pane(document, comm):
    rect = '\n    <svg xmlns="http://www.w3.org/2000/svg">\n      <rect x="10" y="10" height="100" width="100"/>\n    </svg>\n    '
    pane = SVG(rect, encode=True)
    model = pane.get_root(document, comm=comm)
    assert pane._models[model.ref['id']][0] is model
    assert model.text.startswith('&lt;img src=&quot;data:image/svg+xml;base64')
    assert b64encode(rect.encode('utf-8')).decode('utf-8') in model.text
    circle = '\n    <svg xmlns="http://www.w3.org/2000/svg" height="100">\n      <circle cx="50" cy="50" r="40" />\n    </svg>\n    '
    pane.object = circle
    assert pane._models[model.ref['id']][0] is model
    assert model.text.startswith('&lt;img src=&quot;data:image/svg+xml;base64')
    assert b64encode(circle.encode('utf-8')).decode('utf-8') in model.text
    pane.encode = False
    assert model.text == escape(circle)
    pane._cleanup(model)
    assert pane._models == {}
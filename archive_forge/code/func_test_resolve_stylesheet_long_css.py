import os
from pathlib import Path
import bokeh
from packaging.version import Version
from panel.config import config, panel_extension as extension
from panel.io.resources import (
from panel.io.state import set_curdoc
from panel.theme.native import Native
from panel.widgets import Button
def test_resolve_stylesheet_long_css():
    cls = Native
    stylesheet = '\n.styled-button {\n    display: inline-block;\n    padding: 10px 20px;\n    font-size: 16px;\n    font-weight: bold;\n    text-align: center;\n    text-decoration: none;\n    background-color: #4CAF50;\n    color: white;\n    border: none;\n    border-radius: 5px;\n    cursor: pointer;\n    transition: background-color 0.3s;\n}\n\n.styled-button:hover {\n    background-color: #45a049;\n}\n'
    assert resolve_stylesheet(cls, stylesheet, '_stylesheets') == stylesheet
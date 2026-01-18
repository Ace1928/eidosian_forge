import sys
import pytest
from playwright.sync_api import expect
from panel import depends
from panel.layout import Column
from panel.pane import HTML
from panel.tests.util import serve_component, wait_until
from panel.widgets import RadioButtonGroup, TextEditor
def test_texteditor_no_toolbar(page):
    widget = TextEditor(toolbar=False)
    serve_component(page, widget)
    shadowdivs = page.locator('.bk-panel-models-quill-QuillInput > div')
    expect(shadowdivs).to_have_count(1)
    expect(page.locator('.ql-container')).to_be_visible()
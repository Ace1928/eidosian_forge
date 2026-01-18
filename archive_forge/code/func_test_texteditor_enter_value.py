import sys
import pytest
from playwright.sync_api import expect
from panel import depends
from panel.layout import Column
from panel.pane import HTML
from panel.tests.util import serve_component, wait_until
from panel.widgets import RadioButtonGroup, TextEditor
def test_texteditor_enter_value(page):
    widget = TextEditor()
    serve_component(page, widget)
    editor = page.locator('.ql-editor')
    editor.fill('test')
    expect(page.locator('.ql-container')).to_have_text('test')
    wait_until(lambda: widget.value == '<p>test</p>', page)
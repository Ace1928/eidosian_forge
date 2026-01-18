import sys
import pytest
from playwright.sync_api import expect
from panel import depends
from panel.layout import Column
from panel.pane import HTML
from panel.tests.util import serve_component, wait_until
from panel.widgets import RadioButtonGroup, TextEditor
def test_texteditor_regression_click_toolbar_cursor_stays_in_place(page):
    widget = TextEditor()
    serve_component(page, widget)
    editor = page.locator('.ql-editor')
    editor.press('A')
    editor.press('Enter')
    page.locator('.ql-bold').click()
    editor.press('B')
    wait_until(lambda: widget.value == '<p>A</p><p><strong>B</strong></p>', page)
from html import escape
import pytest
from playwright.sync_api import expect
from panel.models import HTML
from panel.pane import Markdown
from panel.tests.util import serve_component, wait_until
def test_update_markdown_pane_empty(page):
    md = Markdown('Initial')
    serve_component(page, md)
    expect(page.locator('.markdown').locator('div')).to_have_text('Initial\n')
    md.object = ''
    expect(page.locator('.markdown').locator('div')).to_have_text('')
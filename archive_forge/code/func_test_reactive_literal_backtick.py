import param
import pytest
from playwright.sync_api import expect
from panel.reactive import ReactiveHTML
from panel.tests.util import serve_component, wait_until
def test_reactive_literal_backtick(page):
    component = ReactiveLiteral(value='Backtick: `')
    serve_component(page, component)
    expect(page.locator('.reactive')).to_have_text('Backtick: `')
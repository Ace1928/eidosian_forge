import pytest
from playwright.sync_api import expect
from panel import config, state
from panel.template import BootstrapTemplate
from panel.tests.util import serve_component
def test_server_reuse_sessions(page, reuse_sessions):

    def app(counts=[0]):
        content = f'### Count {counts[0]}'
        counts[0] += 1
        return content
    _, port = serve_component(page, app)
    expect(page.locator('.markdown h3')).to_have_text('Count 0')
    page.goto(f'http://localhost:{port}')
    expect(page.locator('.markdown h3')).to_have_text('Count 1')
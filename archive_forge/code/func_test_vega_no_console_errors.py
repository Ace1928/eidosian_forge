import pytest
from playwright.sync_api import expect
from panel.pane import Vega
from panel.tests.util import serve_component, wait_until
def test_vega_no_console_errors(page):
    vega = Vega(vega_example)
    msgs, _ = serve_component(page, vega)
    page.wait_for_timeout(1000)
    assert [msg for msg in msgs if msg.type == 'error' and 'favicon' not in msg.location['url']] == []
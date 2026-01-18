import pytest
import numpy as np
from playwright.sync_api import expect
from panel.pane import Vizzu
from panel.tests.util import serve_component, wait_until
def test_vizzu_no_console_errors(page):
    vizzu = Vizzu(DATA, config=CONFIG, duration=400, height=400, sizing_mode='stretch_width', tooltip=True)
    msgs, _ = serve_component(page, vizzu)
    expect(page.locator('canvas')).to_have_count(1)
    bbox = page.locator('canvas').bounding_box()
    assert bbox['width'] > 0
    assert bbox['height'] == 400
    assert [msg for msg in msgs if msg.type == 'error' and 'favicon' not in msg.location['url']] == []
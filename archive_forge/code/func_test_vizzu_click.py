import pytest
import numpy as np
from playwright.sync_api import expect
from panel.pane import Vizzu
from panel.tests.util import serve_component, wait_until
def test_vizzu_click(page):
    vizzu = Vizzu(DATA, config=CONFIG, duration=400, height=400, sizing_mode='stretch_width', tooltip=True)
    clicks = []
    vizzu.on_click(clicks.append)
    msgs, _ = serve_component(page, vizzu)
    expect(page.locator('canvas')).to_have_count(1)
    bbox = page.locator('canvas').bounding_box()
    page.mouse.click(bbox['width'] // 2, bbox['height'] - 100)
    wait_until(lambda: len(clicks) == 1, page)
    assert clicks[0]['categories'] == {'Name': 'Patrick'}
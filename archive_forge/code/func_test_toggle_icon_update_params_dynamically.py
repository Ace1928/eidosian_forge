import pytest
from panel.layout import Row
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
def test_toggle_icon_update_params_dynamically(page):
    icon = ToggleIcon(icon='thumb-down', active_icon='thumb-up')
    serve_component(page, icon)
    assert icon.icon == 'thumb-down'
    assert not icon.value
    assert page.locator('.thumb-down')
    events = []

    def cb(event):
        events.append(event)
    icon.param.watch(cb, 'value')
    page.click('.bk-TablerIcon')
    wait_until(lambda: len(events) == 1, page)
    assert icon.value
    assert page.locator('.thumb-up')
    icon.active_icon = ''
    assert page.locator('.heart-filled')
    icon.value = True
    icon.icon = 'heart'
    assert page.locator('.ti-heart')
    icon.active_icon = ACTIVE_SVG
    assert page.locator('.icon-tabler-ad-filled')
    icon.size = '8em'
    assert page.locator('.icon-tabler-ad-filled').bounding_box()['width'] >= 96
    icon.size = '1em'
    assert page.locator('.icon-tabler-ad-filled').bounding_box()['width'] <= 24
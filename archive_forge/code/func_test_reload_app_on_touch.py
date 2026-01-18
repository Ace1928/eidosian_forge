import os
import pathlib
import time
import pytest
from panel.io.state import state
from panel.tests.util import serve_component
@pytest.mark.parametrize('app', [str(CURPATH / 'app.py'), str(CURPATH / 'app.md'), str(CURPATH / 'app.ipynb')])
def test_reload_app_on_touch(page, autoreload, app):
    path = pathlib.Path(app)
    autoreload(path)
    state.cache['num'] = 0
    serve_component(page, path)
    expect(page.locator('.counter')).to_have_text('0')
    state.cache['num'] = 1
    path.touch()
    expect(page.locator('.counter')).to_have_text('1')
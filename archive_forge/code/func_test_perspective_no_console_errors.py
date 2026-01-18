import pandas as pd
import pytest
from panel.pane import Perspective
from panel.tests.util import serve_component, wait_until
def test_perspective_no_console_errors(page, perspective_data):
    perspective = Perspective(perspective_data)
    msgs, _ = serve_component(page, perspective)
    page.wait_for_timeout(1000)
    assert [msg for msg in msgs if msg.type == 'error' and 'favicon' not in msg.location['url']] == []
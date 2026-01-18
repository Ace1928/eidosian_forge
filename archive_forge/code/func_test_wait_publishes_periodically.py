import os
import sys
import threading
import time
import unittest.mock
import pytest
from cherrypy.process import wspbus
@pytest.mark.xfail(CI_ON_MACOS, reason='continuous integration on macOS fails')
def test_wait_publishes_periodically(bus):
    """Test that wait publishes each tick."""
    callback = unittest.mock.MagicMock()
    bus.subscribe('main', callback)

    def set_start():
        time.sleep(0.05)
        bus.start()
    threading.Thread(target=set_start).start()
    bus.wait(bus.states.STARTED, interval=0.01, channel='main')
    assert callback.call_count > 3
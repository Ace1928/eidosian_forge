import logging
import sys
import time
import uuid
import pytest
import panel as pn
@not_windows
@not_osx
@pytest.mark.subprocess
def test_run_list_args():
    terminal = pn.widgets.Terminal()
    subprocess = terminal.subprocess
    subprocess.args = ['ls', '-l']
    subprocess.run()
    count = 0
    while not subprocess.running and count < 10:
        time.sleep(0.1)
        count += 1
    assert subprocess.running
    subprocess.kill()
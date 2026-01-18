import logging
import sys
import time
import uuid
import pytest
import panel as pn
@not_windows
@not_osx
@pytest.mark.subprocess
def test_subprocess():
    args = 'bash'
    terminal = pn.widgets.Terminal()
    subprocess = terminal.subprocess
    subprocess.args = args
    assert subprocess._terminal == terminal
    assert subprocess.args == args
    assert not subprocess.running
    assert repr(subprocess).startswith('TerminalSubprocess(')
    subprocess.run()
    assert subprocess.running
    assert subprocess._child_pid
    assert subprocess._fd
    subprocess.kill()
    assert not subprocess.running
    assert subprocess._child_pid == 0
    assert subprocess._fd == 0
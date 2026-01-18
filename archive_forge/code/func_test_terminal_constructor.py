import logging
import sys
import time
import uuid
import pytest
import panel as pn
def test_terminal_constructor():
    terminal = pn.widgets.Terminal()
    terminal.write('Hello')
    assert repr(terminal).startswith('Terminal(')
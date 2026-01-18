import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_output_clears_terminal(self):
    clear_calls = []
    uif = ui_testing.TextUIFactory()
    uif.clear_term = lambda: clear_calls.append('clear')
    stream = _mod_ui_text.TextUIOutputStream(uif, uif.stdout, 'utf-8', 'strict')
    stream.write('Hello world!\n')
    stream.write("there's more...\n")
    stream.writelines(['1\n', '2\n', '3\n'])
    self.assertEqual(uif.stdout.getvalue(), "Hello world!\nthere's more...\n1\n2\n3\n")
    self.assertEqual(['clear', 'clear', 'clear'], clear_calls)
    stream.flush()
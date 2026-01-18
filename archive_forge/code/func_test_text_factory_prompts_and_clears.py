import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_factory_prompts_and_clears(self):
    out = ui_testing.StringIOAsTTY()
    self.overrideEnv('TERM', 'xterm')
    factory = ui_testing.TextUIFactory('yada\ny\n', stdout=out, stderr=out)
    with factory:
        pb = factory.nested_progress_bar()
        pb._avail_width = lambda: 79
        pb.show_bar = False
        pb.show_spinner = False
        pb.show_count = False
        pb.update('foo', 0, 1)
        self.assertEqual(True, self.apply_redirected(None, factory.stdout, factory.stdout, factory.get_boolean, 'what do you want'))
        output = out.getvalue()
        self.assertContainsRe(output, '| foo *\r\r  *\r*')
        self.assertContainsString(output, 'what do you want? ([y]es, [n]o): what do you want? ([y]es, [n]o): ')
        self.assertEqual('', factory.stdin.readline())
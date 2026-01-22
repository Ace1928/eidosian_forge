import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
class TestBoolFromString(tests.TestCase):

    def assertIsTrue(self, s, accepted_values=None):
        res = _mod_ui.bool_from_string(s, accepted_values=accepted_values)
        self.assertEqual(True, res)

    def assertIsFalse(self, s, accepted_values=None):
        res = _mod_ui.bool_from_string(s, accepted_values=accepted_values)
        self.assertEqual(False, res)

    def assertIsNone(self, s, accepted_values=None):
        res = _mod_ui.bool_from_string(s, accepted_values=accepted_values)
        self.assertIs(None, res)

    def test_know_valid_values(self):
        self.assertIsTrue('true')
        self.assertIsFalse('false')
        self.assertIsTrue('1')
        self.assertIsFalse('0')
        self.assertIsTrue('on')
        self.assertIsFalse('off')
        self.assertIsTrue('yes')
        self.assertIsFalse('no')
        self.assertIsTrue('y')
        self.assertIsFalse('n')
        self.assertIsTrue('True')
        self.assertIsFalse('False')
        self.assertIsTrue('On')
        self.assertIsFalse('Off')
        self.assertIsTrue('ON')
        self.assertIsFalse('OFF')
        self.assertIsTrue('oN')
        self.assertIsFalse('oFf')

    def test_invalid_values(self):
        self.assertIsNone(None)
        self.assertIsNone('doubt')
        self.assertIsNone('frue')
        self.assertIsNone('talse')
        self.assertIsNone('42')

    def test_provided_values(self):
        av = dict(y=True, n=False, yes=True, no=False)
        self.assertIsTrue('y', av)
        self.assertIsTrue('Y', av)
        self.assertIsTrue('Yes', av)
        self.assertIsFalse('n', av)
        self.assertIsFalse('N', av)
        self.assertIsFalse('No', av)
        self.assertIsNone('1', av)
        self.assertIsNone('0', av)
        self.assertIsNone('on', av)
        self.assertIsNone('off', av)
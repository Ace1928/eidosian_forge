import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_option_callback_bool(self):
    """Test booleans get True and False passed correctly to a callback."""
    cb_calls = []

    def cb(option, name, value, parser):
        cb_calls.append((option, name, value, parser))
    options = [option.Option('hello', custom_callback=cb)]
    opts, args = self.parse(options, ['--hello', '--no-hello'])
    self.assertEqual(2, len(cb_calls))
    opt, name, value, parser = cb_calls[0]
    self.assertEqual('hello', name)
    self.assertTrue(value)
    opt, name, value, parser = cb_calls[1]
    self.assertEqual('hello', name)
    self.assertFalse(value)
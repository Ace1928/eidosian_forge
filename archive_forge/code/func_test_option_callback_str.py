import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_option_callback_str(self):
    """Test callbacks work for string options both long and short."""
    cb_calls = []

    def cb(option, name, value, parser):
        cb_calls.append((option, name, value, parser))
    options = [option.Option('hello', type=str, custom_callback=cb, short_name='h')]
    opts, args = self.parse(options, ['--hello', 'world', '-h', 'mars'])
    self.assertEqual(2, len(cb_calls))
    opt, name, value, parser = cb_calls[0]
    self.assertEqual('hello', name)
    self.assertEqual('world', value)
    opt, name, value, parser = cb_calls[1]
    self.assertEqual('hello', name)
    self.assertEqual('mars', value)
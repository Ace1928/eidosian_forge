import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_skipped_on_HelpCommandIndex_get_topics(self):
    self.hook_missing()
    topic = commands.HelpCommandIndex()
    topics = topic.get_topics('foo')
    self.assertEqual([], self.hook_calls)
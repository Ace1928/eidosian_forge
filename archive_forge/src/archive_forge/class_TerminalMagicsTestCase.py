import sys
import unittest
import os
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from IPython.testing import tools as tt
from IPython.terminal.ptutils import _elide, _adjust_completion_text_based_on_context
from IPython.terminal.shortcuts.auto_suggest import NavigableAutoSuggestFromHistory
class TerminalMagicsTestCase(unittest.TestCase):

    def test_paste_magics_blankline(self):
        """Test that code with a blank line doesn't get split (gh-3246)."""
        ip = get_ipython()
        s = 'def pasted_func(a):\n    b = a+1\n\n    return b'
        tm = ip.magics_manager.registry['TerminalMagics']
        tm.store_or_execute(s, name=None)
        self.assertEqual(ip.user_ns['pasted_func'](54), 55)
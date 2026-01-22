import sys
import unittest
import os
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from IPython.testing import tools as tt
from IPython.terminal.ptutils import _elide, _adjust_completion_text_based_on_context
from IPython.terminal.shortcuts.auto_suggest import NavigableAutoSuggestFromHistory
class mock_input_helper(object):
    """Machinery for tests of the main interact loop.

    Used by the mock_input decorator.
    """

    def __init__(self, testgen):
        self.testgen = testgen
        self.exception = None
        self.ip = get_ipython()

    def __enter__(self):
        self.orig_prompt_for_code = self.ip.prompt_for_code
        self.ip.prompt_for_code = self.fake_input
        return self

    def __exit__(self, etype, value, tb):
        self.ip.prompt_for_code = self.orig_prompt_for_code

    def fake_input(self):
        try:
            return next(self.testgen)
        except StopIteration:
            self.ip.keep_running = False
            return u''
        except:
            self.exception = sys.exc_info()
            self.ip.keep_running = False
            return u''
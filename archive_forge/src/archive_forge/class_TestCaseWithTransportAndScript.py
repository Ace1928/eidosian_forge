import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
class TestCaseWithTransportAndScript(tests.TestCaseWithTransport):
    """Helper class to quickly define shell-like tests.

    Can be used as:

    from breezy.tests import script


    class TestBug(script.TestCaseWithTransportAndScript):

        def test_bug_nnnnn(self):
            self.run_script('''
            $ brz init
            $ brz do-this
            # Boom, error
            ''')
    """

    def setUp(self):
        super().setUp()
        self.script_runner = ScriptRunner()
        self.overrideEnv('INSIDE_EMACS', '1')

    def run_script(self, script, null_output_matches_anything=False):
        return self.script_runner.run_script(self, script, null_output_matches_anything=null_output_matches_anything)

    def run_command(self, cmd, input, output, error):
        return self.script_runner.run_command(self, cmd, input, output, error)
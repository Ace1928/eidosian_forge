import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def _run_transcript_tests(self, transcript_paths: List[str]) -> None:
    """Runs transcript tests for provided file(s).

        This is called when either -t is provided on the command line or the transcript_files argument is provided
        during construction of the cmd2.Cmd instance.

        :param transcript_paths: list of transcript test file paths
        """
    import time
    import unittest
    import cmd2
    from .transcript import Cmd2TestCase

    class TestMyAppCase(Cmd2TestCase):
        cmdapp = self
    transcripts_expanded = utils.files_from_glob_patterns(transcript_paths, access=os.R_OK)
    if not transcripts_expanded:
        self.perror('No test files found - nothing to test')
        self.exit_code = 1
        return
    verinfo = '.'.join(map(str, sys.version_info[:3]))
    num_transcripts = len(transcripts_expanded)
    plural = '' if len(transcripts_expanded) == 1 else 's'
    self.poutput(ansi.style(utils.align_center(' cmd2 transcript test ', fill_char='='), bold=True))
    self.poutput(f'platform {sys.platform} -- Python {verinfo}, cmd2-{cmd2.__version__}, readline-{rl_type}')
    self.poutput(f'cwd: {os.getcwd()}')
    self.poutput(f'cmd2 app: {sys.argv[0]}')
    self.poutput(ansi.style(f'collected {num_transcripts} transcript{plural}', bold=True))
    setattr(self.__class__, 'testfiles', transcripts_expanded)
    sys.argv = [sys.argv[0]]
    testcase = TestMyAppCase()
    stream = cast(TextIO, utils.StdSim(sys.stderr))
    runner = unittest.TextTestRunner(stream=stream)
    start_time = time.time()
    test_results = runner.run(testcase)
    execution_time = time.time() - start_time
    if test_results.wasSuccessful():
        ansi.style_aware_write(sys.stderr, stream.read())
        finish_msg = f' {num_transcripts} transcript{plural} passed in {execution_time:.3f} seconds '
        finish_msg = ansi.style_success(utils.align_center(finish_msg, fill_char='='))
        self.poutput(finish_msg)
    else:
        error_str = stream.read()
        end_of_trace = error_str.find('AssertionError:')
        file_offset = error_str[end_of_trace:].find('File ')
        start = end_of_trace + file_offset
        self.perror(error_str[start:])
        self.exit_code = 1
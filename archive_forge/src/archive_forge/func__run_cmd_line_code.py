import glob
from itertools import chain
import os
import sys
from traitlets.config.application import boolean_flag
from traitlets.config.configurable import Configurable
from traitlets.config.loader import Config
from IPython.core.application import SYSTEM_CONFIG_DIRS, ENV_CONFIG_DIRS
from IPython.core import pylabtools
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import filefind
from traitlets import (
from IPython.terminal import pt_inputhooks
def _run_cmd_line_code(self):
    """Run code or file specified at the command-line"""
    if self.code_to_run:
        line = self.code_to_run
        try:
            self.log.info('Running code given at command line (c=): %s' % line)
            self.shell.run_cell(line, store_history=False)
        except:
            self.log.warning('Error in executing line in user namespace: %s' % line)
            self.shell.showtraceback()
            if not self.interact:
                self.exit(1)
    elif self.file_to_run:
        fname = self.file_to_run
        if os.path.isdir(fname):
            fname = os.path.join(fname, '__main__.py')
        if not os.path.exists(fname):
            self.log.warning("File '%s' doesn't exist", fname)
            if not self.interact:
                self.exit(2)
        try:
            self._exec_file(fname, shell_futures=True)
        except:
            self.shell.showtraceback(tb_offset=4)
            if not self.interact:
                self.exit(1)
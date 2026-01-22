from collections import OrderedDict
import numpy as np
import os
import re
import struct
import sys
import time
import logging
class StdoutProgressIndicator(BaseProgressIndicator):
    """StdoutProgressIndicator(name)

    A progress indicator that shows the progress in stdout. It
    assumes that the tty can appropriately deal with backspace
    characters.
    """

    def _start(self):
        self._chars_prefix, self._chars = ('', '')
        if self._action:
            self._chars_prefix = '%s (%s): ' % (self._name, self._action)
        else:
            self._chars_prefix = '%s: ' % self._name
        sys.stdout.write(self._chars_prefix)
        sys.stdout.flush()

    def _update_progress(self, progressText):
        if not progressText:
            i1, i2, i3, i4 = '-\\|/'
            M = {i1: i2, i2: i3, i3: i4, i4: i1}
            progressText = M.get(self._chars, i1)
        delChars = '\x08' * len(self._chars)
        self._chars = progressText
        sys.stdout.write(delChars + self._chars)
        sys.stdout.flush()

    def _stop(self):
        self._chars = self._chars_prefix = ''
        sys.stdout.write('\n')
        sys.stdout.flush()

    def _write(self, message):
        delChars = '\x08' * len(self._chars_prefix + self._chars)
        sys.stdout.write(delChars + '  ' + message + '\n')
        sys.stdout.write(self._chars_prefix + self._chars)
        sys.stdout.flush()
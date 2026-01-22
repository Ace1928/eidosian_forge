from __future__ import unicode_literals
import os
import re
import six
import termios
import tty
from six.moves import range
from ..keys import Keys
from ..key_binding.input_processor import KeyPress
class raw_mode(object):
    """
    ::

        with raw_mode(stdin):
            ''' the pseudo-terminal stdin is now used in raw mode '''

    We ignore errors when executing `tcgetattr` fails.
    """

    def __init__(self, fileno):
        self.fileno = fileno
        try:
            self.attrs_before = termios.tcgetattr(fileno)
        except termios.error:
            self.attrs_before = None

    def __enter__(self):
        try:
            newattr = termios.tcgetattr(self.fileno)
        except termios.error:
            pass
        else:
            newattr[tty.LFLAG] = self._patch_lflag(newattr[tty.LFLAG])
            newattr[tty.IFLAG] = self._patch_iflag(newattr[tty.IFLAG])
            termios.tcsetattr(self.fileno, termios.TCSANOW, newattr)
            os.write(self.fileno, b'\x1b[?1l')

    @classmethod
    def _patch_lflag(cls, attrs):
        return attrs & ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)

    @classmethod
    def _patch_iflag(cls, attrs):
        return attrs & ~(termios.IXON | termios.IXOFF | termios.ICRNL | termios.INLCR | termios.IGNCR)

    def __exit__(self, *a, **kw):
        if self.attrs_before is not None:
            try:
                termios.tcsetattr(self.fileno, termios.TCSANOW, self.attrs_before)
            except termios.error:
                pass
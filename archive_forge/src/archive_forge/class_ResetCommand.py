from __future__ import division
import re
import stat
from .helpers import (
class ResetCommand(ImportCommand):

    def __init__(self, ref, from_):
        ImportCommand.__init__(self, b'reset')
        self.ref = ref
        self.from_ = from_

    def __bytes__(self):
        if self.from_ is None:
            from_line = b''
        else:
            from_line = b'\nfrom ' + self.from_ + b'\n'
        return b'reset ' + self.ref + from_line
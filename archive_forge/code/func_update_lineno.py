from __future__ import unicode_literals
import re
from pybtex.exceptions import PybtexError
from pybtex import py3compat
def update_lineno(self, value):
    num_newlines = value.count('\n') + value.count('\r') - value.count('\r\n')
    self.lineno += num_newlines
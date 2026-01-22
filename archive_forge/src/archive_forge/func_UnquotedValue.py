from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
def UnquotedValue(self):
    if self.lex is ShellTokenType.ARG or self.lex is ShellTokenType.FLAG:
        return UnquoteShell(self.value)
    else:
        return self.value
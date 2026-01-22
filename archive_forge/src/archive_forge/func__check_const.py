import sys, os
import textwrap
def _check_const(self):
    if self.action not in self.CONST_ACTIONS and self.const is not None:
        raise OptionError("'const' must not be supplied for action %r" % self.action, self)
import sys, os
import textwrap
def _check_opt_strings(self, opts):
    opts = [opt for opt in opts if opt]
    if not opts:
        raise TypeError('at least one option string must be supplied')
    return opts
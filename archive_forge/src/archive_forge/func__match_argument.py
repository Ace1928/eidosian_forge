import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _match_argument(self, action, arg_strings_pattern):
    nargs_pattern = self._get_nargs_pattern(action)
    match = _re.match(nargs_pattern, arg_strings_pattern)
    if match is None:
        nargs_errors = {None: _('expected one argument'), OPTIONAL: _('expected at most one argument'), ONE_OR_MORE: _('expected at least one argument')}
        default = _('expected %s argument(s)') % action.nargs
        msg = nargs_errors.get(action.nargs, default)
        raise ArgumentError(action, msg)
    return len(match.group(1))
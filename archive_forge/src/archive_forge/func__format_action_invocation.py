import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _format_action_invocation(self, action):
    if not action.option_strings:
        metavar, = self._metavar_formatter(action, action.dest)(1)
        return metavar
    else:
        parts = []
        if action.nargs == 0:
            parts.extend(action.option_strings)
        else:
            default = action.dest.upper()
            args_string = self._format_args(action, default)
            for option_string in action.option_strings:
                parts.append('%s %s' % (option_string, args_string))
        return ', '.join(parts)
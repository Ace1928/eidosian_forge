import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _handle_conflict_error(self, action, conflicting_actions):
    message = _('conflicting option string(s): %s')
    conflict_string = ', '.join([option_string for option_string, action in conflicting_actions])
    raise ArgumentError(action, message % conflict_string)
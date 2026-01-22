import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _handle_conflict_resolve(self, action, conflicting_actions):
    for option_string, action in conflicting_actions:
        action.option_strings.remove(option_string)
        self._option_string_actions.pop(option_string, None)
        if not action.option_strings:
            action.container._remove_action(action)
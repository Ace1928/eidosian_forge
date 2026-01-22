import os as _os
import re as _re
import sys as _sys
import warnings
from gettext import gettext as _, ngettext
def _get_default_metavar_for_positional(self, action):
    return action.type.__name__
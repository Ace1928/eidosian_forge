import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _remove_action(self, action):
    self._container._remove_action(action)
    self._group_actions.remove(action)
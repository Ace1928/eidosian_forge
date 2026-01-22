import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def add_argument_group(self, *args, **kwargs):
    group = _ArgumentGroup(self, *args, **kwargs)
    self._action_groups.append(group)
    return group
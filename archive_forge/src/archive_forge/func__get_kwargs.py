import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _get_kwargs(self):
    names = ['prog', 'usage', 'description', 'version', 'formatter_class', 'conflict_handler', 'add_help']
    return [(name, getattr(self, name)) for name in names]
import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _dedent(self):
    self._current_indent -= self._indent_increment
    assert self._current_indent >= 0, 'Indent decreased below 0.'
    self._level -= 1
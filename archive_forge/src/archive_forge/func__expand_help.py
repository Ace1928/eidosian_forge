import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _expand_help(self, action):
    params = dict(vars(action), prog=self._prog)
    for name in list(params):
        if params[name] is SUPPRESS:
            del params[name]
    for name in list(params):
        if hasattr(params[name], '__name__'):
            params[name] = params[name].__name__
    if params.get('choices') is not None:
        choices_str = ', '.join([str(c) for c in params['choices']])
        params['choices'] = choices_str
    return self._get_help_string(action) % params
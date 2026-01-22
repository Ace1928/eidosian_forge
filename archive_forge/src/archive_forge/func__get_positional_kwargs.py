import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _get_positional_kwargs(self, dest, **kwargs):
    if 'required' in kwargs:
        msg = _("'required' is an invalid argument for positionals")
        raise TypeError(msg)
    if kwargs.get('nargs') not in [OPTIONAL, ZERO_OR_MORE]:
        kwargs['required'] = True
    if kwargs.get('nargs') == ZERO_OR_MORE and 'default' not in kwargs:
        kwargs['required'] = True
    return dict(kwargs, dest=dest, option_strings=[])
import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
class _StoreFalseAction(_StoreConstAction):

    def __init__(self, option_strings, dest, default=True, required=False, help=None):
        super(_StoreFalseAction, self).__init__(option_strings=option_strings, dest=dest, const=False, default=default, required=required, help=help)
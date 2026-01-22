import functools
import os
import sys
import os.path
from io import StringIO
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt
@property
def _linenos_special_style(self):
    return 'color: %s; background-color: %s; padding-left: 5px; padding-right: 5px;' % (self.style.line_number_special_color, self.style.line_number_special_background_color)
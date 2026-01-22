from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import stat
import sys
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.utils import system_util
from gslib.utils import text_util
def GetSchemeFromUrlString(url_str):
    """Returns scheme component of a URL string."""
    end_scheme_idx = url_str.find('://')
    if end_scheme_idx == -1:
        return 'file'
    else:
        return url_str[0:end_scheme_idx].lower()
import sys
import os
import re
from email import message_from_file
from distutils.errors import *
from distutils.fancy_getopt import FancyGetopt, translate_longopt
from distutils.util import check_environ, strtobool, rfc822_escape
from distutils import log
from distutils.debug import DEBUG
def _ensure_list(value, fieldname):
    if isinstance(value, str):
        pass
    elif not isinstance(value, list):
        typename = type(value).__name__
        msg = f"Warning: '{fieldname}' should be a list, got type '{typename}'"
        log.log(log.WARN, msg)
        value = list(value)
    return value
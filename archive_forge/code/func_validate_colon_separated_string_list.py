import os
import os.path
import sys
import warnings
import configparser as CP
import codecs
import optparse
from optparse import SUPPRESS_HELP
import docutils
import docutils.utils
import docutils.nodes
from docutils.utils.error_reporting import (locale_encoding, SafeString,
def validate_colon_separated_string_list(setting, value, option_parser, config_parser=None, config_section=None):
    if not isinstance(value, list):
        value = value.split(':')
    else:
        last = value.pop()
        value.extend(last.split(':'))
    return value
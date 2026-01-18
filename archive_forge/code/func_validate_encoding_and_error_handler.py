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
def validate_encoding_and_error_handler(setting, value, option_parser, config_parser=None, config_section=None):
    """
    Side-effect: if an error handler is included in the value, it is inserted
    into the appropriate place as if it was a separate setting/option.
    """
    if ':' in value:
        encoding, handler = value.split(':')
        validate_encoding_error_handler(setting + '_error_handler', handler, option_parser, config_parser, config_section)
        if config_parser:
            config_parser.set(config_section, setting + '_error_handler', handler)
        else:
            setattr(option_parser.values, setting + '_error_handler', handler)
    else:
        encoding = value
    validate_encoding(setting, encoding, option_parser, config_parser, config_section)
    return encoding
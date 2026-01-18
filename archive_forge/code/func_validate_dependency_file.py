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
def validate_dependency_file(setting, value, option_parser, config_parser=None, config_section=None):
    try:
        return docutils.utils.DependencyList(value)
    except IOError:
        return docutils.utils.DependencyList(None)
from __future__ import unicode_literals
import optparse
import sys
import six
from pybtex import __version__, errors
from pybtex.plugin import enumerate_plugin_names, find_plugin
from pybtex.textutils import add_period
class PybtexOption(optparse.Option):
    ATTRS = optparse.Option.ATTRS + ['plugin_group']
    TYPES = optparse.Option.TYPES + ('load_plugin',)
    TYPE_CHECKER = dict(optparse.Option.TYPE_CHECKER, load_plugin=check_plugin)
    STANDARD_OPTIONS = {}
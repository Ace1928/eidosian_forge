from __future__ import unicode_literals
import optparse
import sys
import six
from pybtex import __version__, errors
from pybtex.plugin import enumerate_plugin_names, find_plugin
from pybtex.textutils import add_period
class PybtexHelpFormatter(BaseHelpFormatter):

    def get_plugin_choices(self, plugin_group):
        return ', '.join(sorted(enumerate_plugin_names(plugin_group)))

    def expand_default(self, option):
        result = BaseHelpFormatter.expand_default(self, option)
        if option.plugin_group:
            plugin_choices = self.get_plugin_choices(option.plugin_group)
            result = result.replace('%plugin_choices', plugin_choices)
        return result
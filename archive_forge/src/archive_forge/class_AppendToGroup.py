import argparse
import collections
from osc_lib.command import command
from osc_lib import exceptions
from osc_placement.resources import common
from osc_placement import version
class AppendToGroup(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, '_current_group', None) is None:
            groups = namespace.__dict__.setdefault('groups', {})
            namespace._current_group = ''
            groups[''] = collections.defaultdict(list)
        namespace.groups[namespace._current_group][self.dest].append(values)
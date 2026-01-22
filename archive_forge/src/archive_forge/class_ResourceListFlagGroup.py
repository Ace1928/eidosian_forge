from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
class ResourceListFlagGroup(BinaryCommandFlag):
    """Encapsulates create/set/update/remove key-value flags."""

    def __init__(self, name, help=None, help_name=None, set_flag_only=False):
        """Create a new resource list flag group.

    Args:
      name: the name to be used in the flag names (e.g. "config-maps")
      help: supplementary help text that explains the key-value pairs
      help_name: the resource name to use in help text if different from `name`
      set_flag_only: whether to just add the set-{name} flag.
    """
        super(ResourceListFlagGroup, self).__init__()
        self.set_flag_only = set_flag_only
        self.help = help if not help else help + '\n\n'
        help_name = name if help_name is None else help_name
        pairs_help = 'List of key-value pairs to set as {}.'.format(help_name)
        set_help = pairs_help
        if set_flag_only:
            if help:
                set_help += '\n\n' + help
        else:
            set_help += ' All existing {} will be removed first.'.format(help_name)
        self.clear_flag = BasicFlag('--clear-{}'.format(name), help='If true, removes all {}.'.format(help_name))
        self.set_flag = StringListFlag('--set-{}'.format(name), metavar='KEY=VALUE', help=set_help)
        self.remove_flag = StringListFlag('--remove-{}'.format(name), metavar='KEY', help='List of {} to be removed.'.format(help_name))
        update_aliases = []
        if name == 'labels':
            update_aliases.append('--labels')
        self.update_flag = StringListFlag('--update-{}'.format(name), *update_aliases, metavar='KEY=VALUE', help=pairs_help)

    def AddToParser(self, parser):
        if self.set_flag_only:
            self.set_flag.AddToParser(parser)
            return
        mutex_group = parser.add_mutually_exclusive_group(help=self.help)
        self.clear_flag.AddToParser(mutex_group)
        self.set_flag.AddToParser(mutex_group)
        update_group = mutex_group.add_group(help='Only `{update}` and `{remove}` can be used together. If both are specified, `{remove}` will be applied first.'.format(update=self.update_flag.arg.name, remove=self.remove_flag.arg.name))
        self.remove_flag.AddToParser(update_group)
        self.update_flag.AddToParser(update_group)

    def FormatFlags(self, args):
        if self.set_flag_only:
            return self.set_flag.FormatFlags(args)
        return self.clear_flag.FormatFlags(args) + self.set_flag.FormatFlags(args) + self.remove_flag.FormatFlags(args) + self.update_flag.FormatFlags(args)
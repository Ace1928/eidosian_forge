from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.config import completers
from googlecloudsdk.core import log
from googlecloudsdk.core.configurations import named_configs
class Rename(base.SilentCommand):
    """Renames a named configuration."""
    detailed_help = {'DESCRIPTION': '          {description}\n\n          See `gcloud topic configurations` for an overview of named\n          configurations.\n          ', 'EXAMPLES': '          To rename an existing configuration named `my-config`, run:\n\n            $ {command} my-config --new-name=new-config\n          '}

    @staticmethod
    def Args(parser):
        """Adds args for this command."""
        parser.add_argument('configuration_name', completer=completers.NamedConfigCompleter, help='Name of the configuration to rename')
        parser.add_argument('--new-name', required=True, help='Specifies the new name of the configuration.')

    def Run(self, args):
        named_configs.ConfigurationStore.RenameConfig(args.configuration_name, args.new_name)
        log.status.Print('Renamed [{0}] to be [{1}].'.format(args.configuration_name, args.new_name))
        return args.new_name
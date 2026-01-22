from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import walker_util
def WalkCommandTree(command, prefix):
    """Visit each command and group in the CLI command tree.

    Args:
      command: dict, The tree (nested dict) of command/group names.
      prefix: [str], The subcommand arg prefix.
    """
    name = command.get(_LOOKUP_INTERNAL_NAME)
    args = prefix + [name]
    commands = command.get(cli_tree.LOOKUP_COMMANDS, [])
    groups = command.get(cli_tree.LOOKUP_GROUPS, [])
    names = []
    for c in commands + groups:
        names.append(c.get(_LOOKUP_INTERNAL_NAME, c))
    if names:
        flags = command.get(_LOOKUP_INTERNAL_FLAGS, [])
        if prefix:
            out.write('{identifier}=({args})\n'.format(identifier=ConvertPathToIdentifier(args), args=' '.join(names + flags)))
        else:
            out.write('{identifier}=({args})\n'.format(identifier=ConvertPathToIdentifier(['-GCLOUD-WIDE-FLAGS-']), args=' '.join(flags)))
            out.write('{identifier}=({args})\n'.format(identifier=ConvertPathToIdentifier(args), args=' '.join(names)))
        for c in commands:
            name = c.get(_LOOKUP_INTERNAL_NAME, c)
            flags = c.get(_LOOKUP_INTERNAL_FLAGS, [])
            out.write('{identifier}=({args})\n'.format(identifier=ConvertPathToIdentifier(args + [name]), args=' '.join(flags)))
    for g in groups:
        WalkCommandTree(g, args)
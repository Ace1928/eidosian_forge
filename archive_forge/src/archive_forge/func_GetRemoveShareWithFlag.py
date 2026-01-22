from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
def GetRemoveShareWithFlag(custom_name=None):
    """Gets the --remove-share-with flag."""
    help_text = '  A list of specific projects to remove from the list of projects that this\n  reservation is shared with. List must contain project IDs or project numbers.\n  '
    return base.Argument(custom_name if custom_name else '--remove-share-with', type=arg_parsers.ArgList(min_length=1), metavar='PROJECT', help=help_text)
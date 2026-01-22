from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def GetTagsUpdateFromArgs(args, tags):
    """Generate the change to the tags on a resource based on the arguments.

  Args:
    args: The args for this method.
    tags: The current list of tags.

  Returns:
    The change to the tags after all of the arguments are applied.
  """
    tags_update = tags
    if args.IsKnownAndSpecified('clear_tags'):
        tags_update = []
    if args.IsKnownAndSpecified('add_tags'):
        tags_update = sorted(set(tags_update + args.add_tags))
    if args.IsKnownAndSpecified('remove_tags'):
        tags_update = sorted(set(tags_update) - set(args.remove_tags))
    return tags_update
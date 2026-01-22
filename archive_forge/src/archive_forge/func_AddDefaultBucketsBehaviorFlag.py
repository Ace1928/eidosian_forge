from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
import six
def AddDefaultBucketsBehaviorFlag(parser):
    """Adds a default buckets behavior flag.

  Args:
    parser: The argparse parser to add the arg to.
  """
    GetDefaultBucketsBehaviorFlagMapper().choice_arg.AddToParser(parser)
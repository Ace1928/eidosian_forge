from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def GetUpdateLabelsDictFromArgs(args):
    """Returns the update labels dict from the parsed args.

  Args:
    args: The parsed args.

  Returns:
    The update labels dict from the parsed args.
  """
    return args.labels if hasattr(args, 'labels') else args.update_labels
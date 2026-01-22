from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def AddCreateLabelsFlags(parser):
    """Adds create command labels flags to an argparse parser.

  Args:
    parser: The argparse parser to add the flags to.
  """
    GetCreateLabelsFlag().AddToParser(parser)
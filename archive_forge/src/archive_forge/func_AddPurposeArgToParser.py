from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddPurposeArgToParser(parser):
    """Adds argument for the TagKey's purpose to the parser.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
  """
    parser.add_argument('--purpose', metavar='PURPOSE', choices=['GCE_FIREWALL'], help='Purpose specifier of the TagKey that can only be set on creation. Specifying this field adds additional validation from the policy system that corresponds to the purpose.')
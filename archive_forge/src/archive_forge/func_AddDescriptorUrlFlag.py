from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
def AddDescriptorUrlFlag(parser):
    """Add the descriptor URL argument.

  Args:
    parser: An argparse parser that you can use to add arguments that go on the
      command line after this command. Positional arguments are allowed.
  """
    parser.add_argument('--descriptor-url', help='URL of API of your type.', required=True)
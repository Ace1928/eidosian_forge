from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.args import common_args
def AddConstraintArgToParser(parser):
    """Adds argument for the constraint name to the parser.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
  """
    parser.add_argument('constraint', metavar='CONSTRAINT', help='Name of the org policy constraint. The list of available constraints can be found here: https://cloud.google.com/resource-manager/docs/organization-policy/org-policy-constraints')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddClearAdminUsers(parser):
    """Adds flag for clearing admin users.

  Args:
    parser: The argparse.parser to add the arguments to.
  """
    parser.add_argument('--clear-admin-users', action='store_true', default=None, help='Clear the admin users associated with the cluster')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddType(parser):
    parser.add_argument('--type', help="Cloud SQL user's type. It determines the method to authenticate the user during login. See the list of user types at https://cloud.google.com/sql/docs/postgres/admin-api/rest/v1beta4/SqlUserType")
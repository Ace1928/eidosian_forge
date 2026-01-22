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
def AddConnectorEnforcement(parser, hidden=False):
    """Adds the '--connector-enforcement' flag to the parser.

  Args:
    parser: The current argparse parser to add this to.
    hidden: if the field needs to be hidden.
  """
    help_text = 'Cloud SQL Connector enforcement mode. It determines how Cloud SQL Connectors are used in the connection. See the list of modes [here](https://cloud.google.com/sql/docs/mysql/admin-api/rest/v1beta4/instances#connectorenforcement).'
    parser.add_argument('--connector-enforcement', choices={'CONNECTOR_ENFORCEMENT_UNSPECIFIED': 'The requirement for Cloud SQL connectors is unknown.', 'NOT_REQUIRED': 'Does not require Cloud SQL connectors.', 'REQUIRED': 'Requires all connections to use Cloud SQL connectors, including the Cloud SQL Auth Proxy and Cloud SQL Java, Python, and Go connectors. Note: This disables all existing authorized networks.'}, required=False, default=None, help=help_text, hidden=hidden)
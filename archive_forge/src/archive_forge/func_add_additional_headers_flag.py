from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def add_additional_headers_flag(parser):
    """Adds a flag that allows users to specify arbitrary headers in API calls."""
    parser.add_argument('--additional-headers', action=actions.StoreProperty(properties.VALUES.storage.additional_headers), metavar='HEADER=VALUE', help='Includes arbitrary headers in storage API calls. Accepts a comma separated list of key=value pairs, e.g. `header1=value1,header2=value2`.')
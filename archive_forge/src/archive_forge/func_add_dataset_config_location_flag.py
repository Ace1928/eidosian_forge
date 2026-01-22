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
def add_dataset_config_location_flag(parser, is_required=True):
    """Adds the location flag for the dataset-config commands.

  Args:
    parser (parser_arguments.ArgumentInterceptor): Parser passed to surface.
    is_required (bool): True if location flag is a required field.
  """
    parser.add_argument('--location', type=str, required=is_required, help='Provide location of the dataset config.')
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
def add_dataset_config_create_update_flags(parser, is_update=False):
    """Adds the flags for the dataset-config create and update commands.

  Args:
    parser (parser_arguments.ArgumentInterceptor): Parser passed to surface.
    is_update (bool): True if flags are for the dataset-configs update command.
  """
    parser.add_argument('--retention-period-days', type=int, metavar='RETENTION_DAYS', required=not is_update, help='Provide retention period for the config.')
    parser.add_argument('--description', type=str, help='Description for dataset config.')
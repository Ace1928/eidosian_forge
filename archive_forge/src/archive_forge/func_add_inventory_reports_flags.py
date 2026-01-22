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
def add_inventory_reports_flags(parser, require_create_flags=False):
    """Adds the flags for the inventory reports create and update commands.

  Args:
    parser (parser_arguments.ArgumentInterceptor): Parser passed to surface.
    require_create_flags (bool): True if create flags should be required.
  """
    report_format_settings = parser.add_group(mutex=True, help='Report format configuration. Any combination of CSV flags is valid as long as the Parquet flag is not present.')
    report_format_settings.add_argument('--parquet', action='store_true', help='Generate reports in parquet format.')
    csv_format_settings = report_format_settings.add_group(help='Flags for setting CSV format options.')
    csv_format_settings.add_argument('--csv-separator', choices=['\\n', '\\r\\n'], type=str, metavar='SEPARATOR', help='Sets the character used to separate the records in the inventory report CSV file. For example, ``\\n``')
    csv_format_settings.add_argument('--csv-delimiter', type=str, metavar='DELIMITER', help='Sets the delimiter that separates the fields in the inventory report CSV file. For example, ``,``')
    csv_format_settings.add_argument('--csv-header', action=arg_parsers.StoreTrueFalseAction, help='Indicates whether or not headers are included in the inventory report CSV file. Default is None.')
    parser.add_argument('--destination', type=str, metavar='DESTINATION_URL', help='Sets the URL of the destination bucket and path where generated reports are stored.' + _get_optional_help_text(require_create_flags, 'destination'))
    parser.add_argument('--display-name', type=str, help='Sets the editable name of the report configuration.')
    parser.add_argument('--schedule-starts', type=arg_parsers.Day.Parse, metavar='START_DATE', help='Sets the date you want to start generating inventory reports. For example, 2022-01-30. Should be tomorrow or later based on UTC timezone.' + _get_optional_help_text(require_create_flags, 'start_date'))
    parser.add_argument('--schedule-repeats', choices=['daily', 'weekly'], metavar='FREQUENCY', default='daily' if require_create_flags else None, type=str, help='Sets how often the inventory report configuration will run.' + _get_optional_help_text(require_create_flags, 'frequency'))
    parser.add_argument('--schedule-repeats-until', type=arg_parsers.Day.Parse, metavar='END_DATE', help='Sets date after which you want to stop generating inventory reports. For example, 2022-03-30.' + _get_optional_help_text(require_create_flags, 'end_date'))
    if require_create_flags:
        add_inventory_reports_metadata_fields_flag(parser, require_create_flags)
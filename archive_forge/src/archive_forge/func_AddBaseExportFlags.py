from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import export_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def AddBaseExportFlags(parser, gz_supported=True, database_required=False, database_help_text=flags.DEFAULT_DATABASE_LIST_EXPORT_HELP_TEXT):
    """Adds the base export flags to the parser.

  Args:
    parser: The current argparse parser to add these flags to.
    gz_supported: Boolean, specifies whether gz compression is supported.
    database_required: Boolean, specifies whether the database flag is required.
    database_help_text: String, specifies the help text for the database flag.
  """
    base.ASYNC_FLAG.AddToParser(parser)
    flags.AddInstanceArgument(parser)
    uri_help_text = 'The path to the file in Google Cloud Storage where the export will be stored. The URI is in the form gs://bucketName/fileName. If the file already exists, the operation fails.'
    if gz_supported:
        uri_help_text = uri_help_text + ' If the filename ends with .gz, the contents are compressed.'
    flags.AddUriArgument(parser, uri_help_text)
    flags.AddDatabaseList(parser, database_help_text, database_required)
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddInstanceFlag(parser, required=False):
    """Adds --instance flag to the given parser."""
    help_text = '    If the source is a Cloud SQL database, use this field to provide the Cloud\n    SQL instance ID of the source.\n  '
    parser.add_argument('--instance', help=help_text, required=required)
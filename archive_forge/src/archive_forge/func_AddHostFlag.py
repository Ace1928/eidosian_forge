from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddHostFlag(parser, required=False):
    """Adds --host flag to the given parser."""
    help_text = '    IP or hostname of the database.\n    When `--psc-service-attachment` is also specified, this field value should be:\n    1. For Cloud SQL PSC enabled instance - the dns_name field (e.g <uid>.<region>.sql.goog.).\n    2. For Cloud SQL PSA instance (vpc peering) - the private ip of the instance.\n  '
    parser.add_argument('--host', help=help_text, required=required)
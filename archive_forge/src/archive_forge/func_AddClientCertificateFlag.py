from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddClientCertificateFlag(parser, required=False):
    """Adds --client-certificate flag to the given parser."""
    help_text = '    x509 PEM-encoded certificate that will be used by the replica to\n    authenticate against the database server. Database Migration Service\n    encrypts the value when storing it.\n  '
    parser.add_argument('--client-certificate', help=help_text, required=required)
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddSecondaryZoneFlag(parser):
    """Adds a --secondary-zone flag to the given parser."""
    help_text = '    Google Cloud Platform zone where the failover Cloud SQL database\n    instance is located. Used when the Cloud SQL database availability type\n    is REGIONAL (i.e. multiple zones / highly available).\n  '
    parser.add_argument('--secondary-zone', help=help_text)
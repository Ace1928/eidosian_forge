from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddMaintenanceWindowEnd(parser):
    parser.add_argument('--maintenance-window-end', help='\n      End time of the recurring cluster maintenance window in the RFC 3339\n      (https://www.ietf.org/rfc/rfc3339.txt) format. E.g.\n      "2021-01-01T00:00:00Z" or "2021-01-01T00:00:00-05:00"\n      ')
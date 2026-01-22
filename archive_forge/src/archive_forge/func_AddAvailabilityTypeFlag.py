from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddAvailabilityTypeFlag(parser):
    """Adds a --availability-type flag to the given parser."""
    help_text = 'Cloud SQL availability type.'
    choices = ['REGIONAL', 'ZONAL']
    parser.add_argument('--availability-type', help=help_text, choices=choices)
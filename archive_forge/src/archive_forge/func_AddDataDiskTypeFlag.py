from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddDataDiskTypeFlag(parser):
    """Adds a --data-disk-type flag to the given parser."""
    help_text = 'Type of storage.'
    choices = ['PD_SSD', 'PD_HDD']
    parser.add_argument('--data-disk-type', help=help_text, choices=choices)
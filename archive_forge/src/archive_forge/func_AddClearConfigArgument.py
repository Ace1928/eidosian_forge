from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def AddClearConfigArgument(parser):
    parser.add_argument('--clear-config', action='store_true', help='Clear module config in Security Command Center')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def AddConfigArgument(parser):
    parser.add_argument('--config', metavar='CONFIG_VALUE', help='Module config in Security Command Center')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddNodeLocation(parser):
    parser.add_argument('--node-location', required=True, help='\n      Google Edge Cloud zone where nodes in this node pool will be created.\n      ')
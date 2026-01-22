from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddMachineFilter(parser):
    parser.add_argument('--machine-filter', help='\n      Only machines matching this filter will be allowed to join the node\n      pool. The filtering language accepts strings like "name=<name>", and is\n      documented in more detail at https://google.aip.dev/160.\n      ')
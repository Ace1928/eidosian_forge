from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddVmOnlyFlag(parser):
    return parser.add_argument('--vm-only', action='store_true', required=False, default=False, help="      Do not allocate a TPU, only allocate a VM (useful if you're not ready to run on a TPU yet).\n      ")
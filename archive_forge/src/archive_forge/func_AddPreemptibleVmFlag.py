from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddPreemptibleVmFlag(parser):
    return parser.add_argument('--preemptible-vm', required=False, action='store_true', default=False, help='      Create a preemptible Compute Engine VM, instead of a normal (non-preemptible) VM.\n        A preemptible VM costs less per hour, but the Compute Engine service can terminate the\n        instance at any time.\n      ')
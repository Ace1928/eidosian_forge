from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddCpuCountFlag(parser):
    """Adds a --cpu-count flag to the given parser."""
    help_text = '    Whole number value indicating how many vCPUs the machine should\n    contain. Each vCPU count corresponds to a N2 high-mem machine:\n    (https://cloud.google.com/compute/docs/general-purpose-machines#n2_machines).\n  '
    parser.add_argument('--cpu-count', help=help_text, type=int, choices=[2, 4, 8, 16, 32, 64], required=True)
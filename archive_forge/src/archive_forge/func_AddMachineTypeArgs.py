from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddMachineTypeArgs(parser):
    return parser.add_argument('--machine-type', default='n1-standard-1', help="      Specifies the machine type used for the Compute Engine VM. To get a\n      list of available machine types, run 'gcloud compute\n      machine-types list'. If unspecified, the default type is n1-standard-1.\n      ")
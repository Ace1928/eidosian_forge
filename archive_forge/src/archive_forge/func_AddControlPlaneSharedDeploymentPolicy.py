from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddControlPlaneSharedDeploymentPolicy(parser):
    parser.add_argument('--control-plane-shared-deployment-policy', help='\n      Policy configuration about how user application is deployed for\n      local control plane cluster. It supports two values, ALLOWED and\n      DISALLOWED. ALLOWED means that user application can be deployed on\n      control plane nodes. DISALLOWED means that user application can not be\n      deployed on control plane nodes. Instead, it can only be deployed on\n      worker nodes. By default, this value is DISALLOWED. The input is case\n      insensitive.\n      ')
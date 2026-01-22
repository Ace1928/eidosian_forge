from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetTokenUserFlag():
    """Anthos auth token user flag, specifies the User used in kubeconfig."""
    return base.Argument('--user', required=False, help='User used in kubeconfig.')
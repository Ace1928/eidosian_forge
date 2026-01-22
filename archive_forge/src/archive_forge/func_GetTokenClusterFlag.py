from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetTokenClusterFlag():
    """Anthos auth token cluster flag, specifies cluster name for creating AWS token."""
    return base.Argument('--cluster', required=False, help='Name of the cluster for which token is created.')
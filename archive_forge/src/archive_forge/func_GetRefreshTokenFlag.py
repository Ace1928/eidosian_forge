from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetRefreshTokenFlag():
    """Anthos auth token refresh-token flag, specifies the Refresh Token received from identity provider after authorization flow."""
    return base.Argument('--refresh-token', required=False, help='Refresh Token received from identity provider after authorization flow.')
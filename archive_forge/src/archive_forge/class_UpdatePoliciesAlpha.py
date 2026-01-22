from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import policies as policies_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import policies
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class UpdatePoliciesAlpha(UpdatePoliciesGA):
    _API_VERSION = 'v1alpha'
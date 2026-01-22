from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_util
from googlecloudsdk.calliope import base
def UpdateGroup(self, group_ref, group, update_mask, request_id=None):
    """Call API to update an existing group."""
    name = group_ref.RelativeName()
    update_mask_string = ','.join(update_mask)
    update_req = self.messages.NetworkconnectivityProjectsLocationsGlobalHubsGroupsPatchRequest(name=name, requestId=request_id, group=group, updateMask=update_mask_string)
    return self.group_service.Patch(update_req)
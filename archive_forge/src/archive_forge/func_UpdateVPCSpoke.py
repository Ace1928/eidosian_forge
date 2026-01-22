from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_util
from googlecloudsdk.calliope import base
def UpdateVPCSpoke(self, spoke_ref, spoke, update_mask, request_id=None):
    """Call API to update a existing spoke."""
    name = spoke_ref.RelativeName()
    update_mask_string = ','.join(update_mask)
    update_req = self.messages.NetworkconnectivityProjectsLocationsSpokesPatchRequest(name=name, requestId=request_id, spoke=spoke, updateMask=update_mask_string)
    return self.spoke_service.Patch(update_req)
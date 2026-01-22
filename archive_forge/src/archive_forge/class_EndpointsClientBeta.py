from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.service_directory import base as sd_base
from googlecloudsdk.calliope import base
class EndpointsClientBeta(EndpointsClient):
    """Client for endpoints in the v1beta1 Service Directory API."""

    def __init__(self):
        super(EndpointsClientBeta, self).__init__(base.ReleaseTrack.BETA)

    def Create(self, endpoint_ref, address=None, port=None, metadata=None, network=None):
        """Endpoints create request."""
        endpoint = self.msgs.Endpoint(address=address, port=port, metadata=metadata, network=network)
        create_req = self.msgs.ServicedirectoryProjectsLocationsNamespacesServicesEndpointsCreateRequest(parent=endpoint_ref.Parent().RelativeName(), endpoint=endpoint, endpointId=endpoint_ref.endpointsId)
        return self.service.Create(create_req)

    def Update(self, endpoint_ref, address=None, port=None, metadata=None):
        """Endpoints update request."""
        mask_parts = []
        if address is not None:
            mask_parts.append('address')
        if port is not None:
            mask_parts.append('port')
        if metadata is not None:
            mask_parts.append('metadata')
        endpoint = self.msgs.Endpoint(address=address, port=port, metadata=metadata)
        update_req = self.msgs.ServicedirectoryProjectsLocationsNamespacesServicesEndpointsPatchRequest(name=endpoint_ref.RelativeName(), endpoint=endpoint, updateMask=','.join(mask_parts))
        return self.service.Patch(update_req)
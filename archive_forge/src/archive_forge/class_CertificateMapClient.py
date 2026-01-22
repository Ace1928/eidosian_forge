from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.certificate_manager import api_client
class CertificateMapClient(object):
    """API client for Certificate Manager certificate maps."""

    def __init__(self, client=None, messages=None):
        self._client = client or api_client.GetClientInstance()
        self._service = self._client.projects_locations_certificateMaps
        self.messages = messages or self._client.MESSAGES_MODULE

    def Create(self, parent_ref, map_id, description='', labels=None):
        """Creates a certificate map.

    Args:
      parent_ref: a Resource reference to a
        certificatemanager.projects.locations resource for the parent of this
        certificate map.
      map_id: str, the ID of the map to create.
      description: str, user-provided description.
      labels: Unified GCP Labels for the resource.

    Returns:
      Operation: the long running operation to create a map.
    """
        req = self.messages.CertificatemanagerProjectsLocationsCertificateMapsCreateRequest(parent=parent_ref.RelativeName(), certificateMapId=map_id, certificateMap=self.messages.CertificateMap(labels=labels, description=description))
        return self._service.Create(req)

    def Get(self, map_ref):
        """Gets certificate map.

    Args:
      map_ref: a Resource reference to a
        certificatemanager.projects.locations.certificateMaps resource to get.

    Returns:
      Certificate Map API representation.
    """
        get_req = self.messages.CertificatemanagerProjectsLocationsCertificateMapsGetRequest(name=map_ref.RelativeName())
        return self._service.Get(get_req)

    def List(self, parent_ref, limit=None, page_size=None, list_filter=None, order_by=None):
        """List certificate maps in a given project and location.

    Args:
      parent_ref: a Resource reference to a
        certificatemanager.projects.locations resource to list maps for.
      limit: int, the total number of results to return from the API.
      page_size: int, the number of results in each batch from the API.
      list_filter: str, filter to apply in the list request.
      order_by: str, fields used for resource ordering.

    Returns:
      A list of the certificate maps in the project.
    """
        list_req = self.messages.CertificatemanagerProjectsLocationsCertificateMapsListRequest(parent=parent_ref.RelativeName(), filter=list_filter, orderBy=order_by)
        return list_pager.YieldFromList(self._service, list_req, batch_size=page_size, limit=limit, current_token_attribute='pageToken', next_token_attribute='nextPageToken', field='certificateMaps', batch_size_attribute='pageSize')

    def Delete(self, map_ref):
        """Deletes certificate map.

    Args:
      map_ref: a Resource reference to a
        certificatemanager.projects.locations.certificateMaps resource to
        delete.

    Returns:
      Operation: the long running operation to delete certificate map.
    """
        delete_req = self.messages.CertificatemanagerProjectsLocationsCertificateMapsDeleteRequest(name=map_ref.RelativeName())
        return self._service.Delete(delete_req)

    def Patch(self, map_ref, labels=None, description=None):
        """Updates a certificate map.

    Used for updating labels and description.

    Args:
      map_ref: a Resource reference to a
        certificatemanager.projects.locations.certificateMaps resource.
      labels: unified GCP Labels for the resource.
      description: str, new description

    Returns:
      Operation: the long running operation to patch certificate map.
    """
        certificate_map = self.messages.CertificateMap()
        updated_fields = []
        if labels:
            certificate_map.labels = labels
            updated_fields.append('labels')
        if description:
            certificate_map.description = description
            updated_fields.append('description')
        update_mask = ','.join(updated_fields)
        patch_req = self.messages.CertificatemanagerProjectsLocationsCertificateMapsPatchRequest(certificateMap=certificate_map, name=map_ref.RelativeName(), updateMask=update_mask)
        return self._service.Patch(patch_req)
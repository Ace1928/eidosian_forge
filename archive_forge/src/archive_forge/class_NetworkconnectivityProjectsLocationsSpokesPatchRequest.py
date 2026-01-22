from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkconnectivityProjectsLocationsSpokesPatchRequest(_messages.Message):
    """A NetworkconnectivityProjectsLocationsSpokesPatchRequest object.

  Fields:
    googleCloudNetworkconnectivityV1betaSpoke: A
      GoogleCloudNetworkconnectivityV1betaSpoke resource to be passed as the
      request body.
    name: Immutable. The name of the spoke. Spoke names must be unique. They
      use the following form:
      `projects/{project_number}/locations/{region}/spokes/{spoke_id}`
    requestId: Optional. A request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server knows to
      ignore the request if it has already been completed. The server
      guarantees that a request doesn't result in creation of duplicate
      commitments for at least 60 minutes. For example, consider a situation
      where you make an initial request and the request times out. If you make
      the request again with the same request ID, the server can check to see
      whether the original operation was received. If it was, the server
      ignores the second request. This behavior prevents clients from
      mistakenly creating duplicate commitments. The request ID must be a
      valid UUID, with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
    updateMask: Optional. In the case of an update to an existing spoke, field
      mask is used to specify the fields to be overwritten. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field is overwritten if it is in the mask. If the user does
      not provide a mask, then all fields are overwritten.
  """
    googleCloudNetworkconnectivityV1betaSpoke = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaSpoke', 1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)
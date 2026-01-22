from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KrmapihostingProjectsLocationsKrmApiHostsPatchRequest(_messages.Message):
    """A KrmapihostingProjectsLocationsKrmApiHostsPatchRequest object.

  Fields:
    krmApiHost: A KrmApiHost resource to be passed as the request body.
    name: Output only. The name of this KrmApiHost resource in the format: 'pr
      ojects/{project_id}/locations/{location}/krmApiHosts/{krm_api_host_id}'.
    requestId: Optional. A unique ID to identify requests. This is unique such
      that if the request is re-tried, the server will know to ignore the
      request if it has already been completed. The server will guarantee that
      for at least 60 minutes after the first request. The request ID must be
      a valid UUID with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
    updateMask: Optional. Field mask is used to specify the fields to be
      overwritten in the KrmApiHost resource by the update. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field will be overwritten if it is in the mask. A request
      must specify at least one path in the field mask. Supported field mask
      values are: - `management_config.standard_management_config.man_block`
  """
    krmApiHost = _messages.MessageField('KrmApiHost', 1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)
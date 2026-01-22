from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataAttributeBindingsDeleteRequest(_messages.Message):
    """A DataplexProjectsLocationsDataAttributeBindingsDeleteRequest object.

  Fields:
    etag: Required. If the client provided etag value does not match the
      current etag value, the DeleteDataAttributeBindingRequest method returns
      an ABORTED error response. Etags must be used when calling the
      DeleteDataAttributeBinding.
    name: Required. The resource name of the DataAttributeBinding: projects/{p
      roject_number}/locations/{location_id}/dataAttributeBindings/{data_attri
      bute_binding_id}
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
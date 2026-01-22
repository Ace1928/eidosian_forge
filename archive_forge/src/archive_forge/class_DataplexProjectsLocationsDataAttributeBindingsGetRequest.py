from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataAttributeBindingsGetRequest(_messages.Message):
    """A DataplexProjectsLocationsDataAttributeBindingsGetRequest object.

  Fields:
    name: Required. The resource name of the DataAttributeBinding: projects/{p
      roject_number}/locations/{location_id}/dataAttributeBindings/{data_attri
      bute_binding_id}
  """
    name = _messages.StringField(1, required=True)
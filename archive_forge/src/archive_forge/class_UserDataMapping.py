from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserDataMapping(_messages.Message):
    """Maps a resource to the associated user and Attributes.

  Fields:
    archiveTime: Output only. Indicates the time when this mapping was
      archived.
    archived: Output only. Indicates whether this mapping is archived.
    dataId: Required. A unique identifier for the mapped resource.
    name: Resource name of the User data mapping, of the form `projects/{proje
      ct_id}/locations/{location_id}/datasets/{dataset_id}/consentStores/{cons
      ent_store_id}/userDataMappings/{user_data_mapping_id}`.
    resourceAttributes: Attributes of the resource. Only explicitly set
      attributes are displayed here. Attribute definitions with defaults set
      implicitly apply to these User data mappings. Attributes listed here
      must be single valued, that is, exactly one value is specified for the
      field "values" in each Attribute.
    userId: Required. User's UUID provided by the client.
  """
    archiveTime = _messages.StringField(1)
    archived = _messages.BooleanField(2)
    dataId = _messages.StringField(3)
    name = _messages.StringField(4)
    resourceAttributes = _messages.MessageField('Attribute', 5, repeated=True)
    userId = _messages.StringField(6)
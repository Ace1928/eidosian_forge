from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookupRequest(_messages.Message):
    """The request for Datastore.Lookup.

  Fields:
    databaseId: The ID of the database against which to make the request.
      '(default)' is not allowed; please use empty string '' to refer the
      default database.
    keys: Required. Keys of entities to look up.
    propertyMask: The properties to return. Defaults to returning all
      properties. If this field is set and an entity has a property not
      referenced in the mask, it will be absent from
      LookupResponse.found.entity.properties. The entity's key is always
      returned.
    readOptions: The options for this lookup request.
  """
    databaseId = _messages.StringField(1)
    keys = _messages.MessageField('Key', 2, repeated=True)
    propertyMask = _messages.MessageField('PropertyMask', 3)
    readOptions = _messages.MessageField('ReadOptions', 4)
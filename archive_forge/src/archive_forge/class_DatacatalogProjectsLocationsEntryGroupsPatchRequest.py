from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsPatchRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsPatchRequest object.

  Fields:
    googleCloudDatacatalogV1EntryGroup: A GoogleCloudDatacatalogV1EntryGroup
      resource to be passed as the request body.
    name: Identifier. The resource name of the entry group in URL format.
      Note: The entry group itself and its child resources might not be stored
      in the location specified in its name.
    updateMask: Names of fields whose values to overwrite on an entry group.
      If this parameter is absent or empty, all modifiable fields are
      overwritten. If such fields are non-required and omitted in the request
      body, their values are emptied.
  """
    googleCloudDatacatalogV1EntryGroup = _messages.MessageField('GoogleCloudDatacatalogV1EntryGroup', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
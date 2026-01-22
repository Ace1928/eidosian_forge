from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsEntriesTagsCreateRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsEntriesTagsCreateRequest
  object.

  Fields:
    googleCloudDatacatalogV1Tag: A GoogleCloudDatacatalogV1Tag resource to be
      passed as the request body.
    parent: Required. The name of the resource to attach this tag to. Tags can
      be attached to entries or entry groups. An entry can have up to 1000
      attached tags. Note: The tag and its child resources might not be stored
      in the location specified in its name.
  """
    googleCloudDatacatalogV1Tag = _messages.MessageField('GoogleCloudDatacatalogV1Tag', 1)
    parent = _messages.StringField(2, required=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsEntriesUnstarRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsEntriesUnstarRequest object.

  Fields:
    googleCloudDatacatalogV1UnstarEntryRequest: A
      GoogleCloudDatacatalogV1UnstarEntryRequest resource to be passed as the
      request body.
    name: Required. The name of the entry to mark as **not** starred.
  """
    googleCloudDatacatalogV1UnstarEntryRequest = _messages.MessageField('GoogleCloudDatacatalogV1UnstarEntryRequest', 1)
    name = _messages.StringField(2, required=True)
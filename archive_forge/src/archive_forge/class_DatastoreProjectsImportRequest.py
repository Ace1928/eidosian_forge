from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastoreProjectsImportRequest(_messages.Message):
    """A DatastoreProjectsImportRequest object.

  Fields:
    googleDatastoreAdminV1ImportEntitiesRequest: A
      GoogleDatastoreAdminV1ImportEntitiesRequest resource to be passed as the
      request body.
    projectId: Required. Project ID against which to make the request.
  """
    googleDatastoreAdminV1ImportEntitiesRequest = _messages.MessageField('GoogleDatastoreAdminV1ImportEntitiesRequest', 1)
    projectId = _messages.StringField(2, required=True)
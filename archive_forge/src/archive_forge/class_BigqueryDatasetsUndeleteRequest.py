from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryDatasetsUndeleteRequest(_messages.Message):
    """A BigqueryDatasetsUndeleteRequest object.

  Fields:
    datasetId: Required. Dataset ID of dataset being deleted
    projectId: Required. Project ID of the dataset to be undeleted
    undeleteDatasetRequest: A UndeleteDatasetRequest resource to be passed as
      the request body.
  """
    datasetId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    undeleteDatasetRequest = _messages.MessageField('UndeleteDatasetRequest', 3)
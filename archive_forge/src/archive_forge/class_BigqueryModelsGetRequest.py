from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryModelsGetRequest(_messages.Message):
    """A BigqueryModelsGetRequest object.

  Fields:
    datasetId: Required. Dataset ID of the requested model.
    modelId: Required. Model ID of the requested model.
    projectId: Required. Project ID of the requested model.
  """
    datasetId = _messages.StringField(1, required=True)
    modelId = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)
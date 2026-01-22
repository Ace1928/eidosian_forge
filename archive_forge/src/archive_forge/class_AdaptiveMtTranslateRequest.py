from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdaptiveMtTranslateRequest(_messages.Message):
    """The request for sending an AdaptiveMt translation query.

  Fields:
    content: Required. The content of the input in string format. For now only
      one sentence per request is supported.
    dataset: Required. The resource name for the dataset to use for adaptive
      MT. `projects/{project}/locations/{location-
      id}/adaptiveMtDatasets/{dataset}`
  """
    content = _messages.StringField(1, repeated=True)
    dataset = _messages.StringField(2)
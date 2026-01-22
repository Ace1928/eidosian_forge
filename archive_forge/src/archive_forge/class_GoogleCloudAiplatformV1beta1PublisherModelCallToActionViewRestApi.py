from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PublisherModelCallToActionViewRestApi(_messages.Message):
    """Rest API docs.

  Fields:
    documentations: Required.
    title: Required. The title of the view rest API.
  """
    documentations = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelDocumentation', 1, repeated=True)
    title = _messages.StringField(2)
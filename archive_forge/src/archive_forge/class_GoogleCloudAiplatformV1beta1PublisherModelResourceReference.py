from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PublisherModelResourceReference(_messages.Message):
    """Reference to a resource.

  Fields:
    description: Description of the resource.
    resourceName: The resource name of the Google Cloud resource.
    uri: The URI of the resource.
    useCase: Use case (CUJ) of the resource.
  """
    description = _messages.StringField(1)
    resourceName = _messages.StringField(2)
    uri = _messages.StringField(3)
    useCase = _messages.StringField(4)
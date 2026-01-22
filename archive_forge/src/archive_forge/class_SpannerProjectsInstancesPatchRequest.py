from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesPatchRequest(_messages.Message):
    """A SpannerProjectsInstancesPatchRequest object.

  Fields:
    name: Required. A unique identifier for the instance, which cannot be
      changed after the instance is created. Values are of the form
      `projects//instances/a-z*[a-z0-9]`. The final segment of the name must
      be between 2 and 64 characters in length.
    updateInstanceRequest: A UpdateInstanceRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    updateInstanceRequest = _messages.MessageField('UpdateInstanceRequest', 2)
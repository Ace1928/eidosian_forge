from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesCreateRequest(_messages.Message):
    """A SpannerProjectsInstancesCreateRequest object.

  Fields:
    createInstanceRequest: A CreateInstanceRequest resource to be passed as
      the request body.
    parent: Required. The name of the project in which to create the instance.
      Values are of the form `projects/`.
  """
    createInstanceRequest = _messages.MessageField('CreateInstanceRequest', 1)
    parent = _messages.StringField(2, required=True)
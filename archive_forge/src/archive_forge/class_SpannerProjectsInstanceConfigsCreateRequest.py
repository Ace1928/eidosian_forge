from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstanceConfigsCreateRequest(_messages.Message):
    """A SpannerProjectsInstanceConfigsCreateRequest object.

  Fields:
    createInstanceConfigRequest: A CreateInstanceConfigRequest resource to be
      passed as the request body.
    parent: Required. The name of the project in which to create the instance
      config. Values are of the form `projects/`.
  """
    createInstanceConfigRequest = _messages.MessageField('CreateInstanceConfigRequest', 1)
    parent = _messages.StringField(2, required=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudresourcemanagerProjectsCreateRequest(_messages.Message):
    """A CloudresourcemanagerProjectsCreateRequest object.

  Fields:
    project: A Project resource to be passed as the request body.
    useLegacyStack: A now unused experiment opt-out option.
  """
    project = _messages.MessageField('Project', 1)
    useLegacyStack = _messages.BooleanField(2)
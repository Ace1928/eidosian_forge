from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudkmsProjectsSetProjectOptOutStateRequest(_messages.Message):
    """A CloudkmsProjectsSetProjectOptOutStateRequest object.

  Fields:
    name: Required. Project number or id for which to set the opt-out
      preference, in the format `projects/123456789` (or `projects/my-
      project`).
    setProjectOptOutStateRequest: A SetProjectOptOutStateRequest resource to
      be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    setProjectOptOutStateRequest = _messages.MessageField('SetProjectOptOutStateRequest', 2)
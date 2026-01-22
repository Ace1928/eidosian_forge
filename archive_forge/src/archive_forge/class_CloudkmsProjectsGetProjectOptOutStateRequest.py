from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudkmsProjectsGetProjectOptOutStateRequest(_messages.Message):
    """A CloudkmsProjectsGetProjectOptOutStateRequest object.

  Fields:
    name: Required. Project number or id for which to get the opt-out
      preference, in the format `projects/123456789` (or `projects/my-
      project`).
  """
    name = _messages.StringField(1, required=True)
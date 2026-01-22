from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerProjectsGetRequest(_messages.Message):
    """A CloudresourcemanagerProjectsGetRequest object.

  Fields:
    projectId: Required. The Project ID (for example, `my-project-123`).
  """
    projectId = _messages.StringField(1, required=True)
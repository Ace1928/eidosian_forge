from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsInstallationsListRequest(_messages.Message):
    """A CloudbuildProjectsInstallationsListRequest object.

  Fields:
    parent: The parent resource where github installations for project will be
      listed. Format: `projects/{project}/locations/{location}`
    projectId: Project id
  """
    parent = _messages.StringField(1)
    projectId = _messages.StringField(2, required=True)
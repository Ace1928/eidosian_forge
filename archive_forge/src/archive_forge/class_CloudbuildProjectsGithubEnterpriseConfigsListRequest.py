from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsGithubEnterpriseConfigsListRequest(_messages.Message):
    """A CloudbuildProjectsGithubEnterpriseConfigsListRequest object.

  Fields:
    parent: Name of the parent project. For example:
      projects/{$project_number} or projects/{$project_id}
    projectId: ID of the project
  """
    parent = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2)
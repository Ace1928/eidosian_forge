from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsBuildsCreateRequest(_messages.Message):
    """A CloudbuildProjectsLocationsBuildsCreateRequest object.

  Fields:
    build: A Build resource to be passed as the request body.
    parent: The parent resource where this build will be created. Format:
      `projects/{project}/locations/{location}`
    projectId: Required. ID of the project.
  """
    build = _messages.MessageField('Build', 1)
    parent = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3)
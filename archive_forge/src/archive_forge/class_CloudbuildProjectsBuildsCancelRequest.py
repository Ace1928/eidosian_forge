from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsBuildsCancelRequest(_messages.Message):
    """A CloudbuildProjectsBuildsCancelRequest object.

  Fields:
    cancelBuildRequest: A CancelBuildRequest resource to be passed as the
      request body.
    id: Required. ID of the build.
    projectId: Required. ID of the project.
  """
    cancelBuildRequest = _messages.MessageField('CancelBuildRequest', 1)
    id = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)
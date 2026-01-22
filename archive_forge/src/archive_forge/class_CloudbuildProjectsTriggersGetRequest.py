from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsTriggersGetRequest(_messages.Message):
    """A CloudbuildProjectsTriggersGetRequest object.

  Fields:
    name: The name of the `Trigger` to retrieve. Format:
      `projects/{project}/locations/{location}/triggers/{trigger}`
    projectId: Required. ID of the project that owns the trigger.
    triggerId: Required. Identifier (`id` or `name`) of the `BuildTrigger` to
      get.
  """
    name = _messages.StringField(1)
    projectId = _messages.StringField(2, required=True)
    triggerId = _messages.StringField(3, required=True)
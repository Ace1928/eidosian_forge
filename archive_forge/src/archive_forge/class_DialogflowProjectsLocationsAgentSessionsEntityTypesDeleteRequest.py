from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentSessionsEntityTypesDeleteRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentSessionsEntityTypesDeleteRequest
  object.

  Fields:
    name: Required. The name of the entity type to delete. Format:
      `projects//agent/sessions//entityTypes/` or
      `projects//agent/environments//users//sessions//entityTypes/`. If
      `Environment ID` is not specified, we assume default 'draft'
      environment. If `User ID` is not specified, we assume default '-' user.
  """
    name = _messages.StringField(1, required=True)
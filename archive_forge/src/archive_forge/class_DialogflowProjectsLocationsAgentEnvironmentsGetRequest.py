from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentEnvironmentsGetRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentEnvironmentsGetRequest object.

  Fields:
    name: Required. The name of the environment. Supported formats: -
      `projects//agent/environments/` -
      `projects//locations//agent/environments/` The environment ID for the
      default environment is `-`.
  """
    name = _messages.StringField(1, required=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EnvironmentSessionStatus(_messages.Message):
    """Status of sessions created for this environment.

  Fields:
    active: Output only. Queries over sessions to mark whether the environment
      is currently active or not
  """
    active = _messages.BooleanField(1)
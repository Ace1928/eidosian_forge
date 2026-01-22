from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomTargetDeployMetadata(_messages.Message):
    """CustomTargetDeployMetadata contains information from a Custom Target
  deploy operation.

  Fields:
    skipMessage: Output only. Skip message provided in the results of a custom
      deploy operation.
  """
    skipMessage = _messages.StringField(1)
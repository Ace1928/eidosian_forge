from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SeedConversionWorkspaceRequest(_messages.Message):
    """Request message for 'SeedConversionWorkspace' request.

  Fields:
    autoCommit: Should the conversion workspace be committed automatically
      after the seed operation.
    destinationConnectionProfile: Optional. Fully qualified (Uri) name of the
      destination connection profile.
    sourceConnectionProfile: Optional. Fully qualified (Uri) name of the
      source connection profile.
  """
    autoCommit = _messages.BooleanField(1)
    destinationConnectionProfile = _messages.StringField(2)
    sourceConnectionProfile = _messages.StringField(3)
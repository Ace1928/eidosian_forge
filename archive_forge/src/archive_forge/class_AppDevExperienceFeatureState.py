from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppDevExperienceFeatureState(_messages.Message):
    """State for App Dev Exp Feature.

  Fields:
    networkingInstallSucceeded: Status of subcomponent that detects configured
      Service Mesh resources.
  """
    networkingInstallSucceeded = _messages.MessageField('Status', 1)
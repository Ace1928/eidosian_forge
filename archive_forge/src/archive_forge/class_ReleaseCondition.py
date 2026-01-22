from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReleaseCondition(_messages.Message):
    """ReleaseCondition contains all conditions relevant to a Release.

  Fields:
    releaseReadyCondition: Details around the Releases's overall status.
    skaffoldSupportedCondition: Details around the support state of the
      release's Skaffold version.
  """
    releaseReadyCondition = _messages.MessageField('ReleaseReadyCondition', 1)
    skaffoldSupportedCondition = _messages.MessageField('SkaffoldSupportedCondition', 2)
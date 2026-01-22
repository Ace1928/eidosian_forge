from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckOnboardingStatusResponse(_messages.Message):
    """Response message for `CheckOnboardingStatus` method.

  Fields:
    findings: List of issues that are preventing PAM from functioning for this
      resource and need to be fixed to complete onboarding. Not all issues
      might be detected and reported.
    serviceAccount: The service account that PAM will use to act on this
      resource.
  """
    findings = _messages.MessageField('Finding', 1, repeated=True)
    serviceAccount = _messages.StringField(2)
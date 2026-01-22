from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1beta1ViolationRemediationInstructionsConsole(_messages.Message):
    """Remediation instructions to resolve violation via cloud console

  Fields:
    additionalLinks: Additional urls for more information about steps
    consoleUris: Link to console page where violations can be resolved
    steps: Steps to resolve violation via cloud console
  """
    additionalLinks = _messages.StringField(1, repeated=True)
    consoleUris = _messages.StringField(2, repeated=True)
    steps = _messages.StringField(3, repeated=True)
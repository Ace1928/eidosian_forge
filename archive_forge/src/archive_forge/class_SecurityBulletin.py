from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityBulletin(_messages.Message):
    """SecurityBulletin are notifications of vulnerabilities of Google
  products.

  Fields:
    bulletinId: ID of the bulletin corresponding to the vulnerability.
    submissionTime: Submission time of this Security Bulletin.
    suggestedUpgradeVersion: This represents a version that the cluster
      receiving this notification should be upgraded to, based on its current
      version. For example, 1.15.0
  """
    bulletinId = _messages.StringField(1)
    submissionTime = _messages.StringField(2)
    suggestedUpgradeVersion = _messages.StringField(3)
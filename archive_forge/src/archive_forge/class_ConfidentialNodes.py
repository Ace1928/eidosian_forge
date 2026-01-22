from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfidentialNodes(_messages.Message):
    """ConfidentialNodes is configuration for the confidential nodes feature,
  which makes nodes run on confidential VMs.

  Fields:
    enabled: Whether Confidential Nodes feature is enabled.
  """
    enabled = _messages.BooleanField(1)
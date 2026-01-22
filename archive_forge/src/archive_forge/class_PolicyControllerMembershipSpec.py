from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyControllerMembershipSpec(_messages.Message):
    """**Policy Controller**: Configuration for a single cluster. Intended to
  parallel the PolicyController CR.

  Fields:
    policyControllerHubConfig: Policy Controller configuration for the
      cluster.
    version: Version of Policy Controller installed.
  """
    policyControllerHubConfig = _messages.MessageField('PolicyControllerHubConfig', 1)
    version = _messages.StringField(2)
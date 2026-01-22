from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectMacsec(_messages.Message):
    """Configuration information for enabling Media Access Control security
  (MACsec) on this Cloud Interconnect connection between Google and your on-
  premises router.

  Fields:
    failOpen: If set to true, the Interconnect connection is configured with a
      should-secure MACsec security policy, that allows the Google router to
      fallback to cleartext traffic if the MKA session cannot be established.
      By default, the Interconnect connection is configured with a must-secure
      security policy that drops all traffic if the MKA session cannot be
      established with your router.
    preSharedKeys: Required. A keychain placeholder describing a set of named
      key objects along with their start times. A MACsec CKN/CAK is generated
      for each key in the key chain. Google router automatically picks the key
      with the most recent startTime when establishing or re-establishing a
      MACsec secure link.
  """
    failOpen = _messages.BooleanField(1)
    preSharedKeys = _messages.MessageField('InterconnectMacsecPreSharedKey', 2, repeated=True)
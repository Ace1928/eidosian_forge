from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Indicator(_messages.Message):
    """Represents what's commonly known as an _indicator of compromise_ (IoC)
  in computer forensics. This is an artifact observed on a network or in an
  operating system that, with high confidence, indicates a computer intrusion.
  For more information, see [Indicator of
  compromise](https://en.wikipedia.org/wiki/Indicator_of_compromise).

  Fields:
    domains: List of domains associated to the Finding.
    ipAddresses: The list of IP addresses that are associated with the
      finding.
    signatures: The list of matched signatures indicating that the given
      process is present in the environment.
    uris: The list of URIs associated to the Findings.
  """
    domains = _messages.StringField(1, repeated=True)
    ipAddresses = _messages.StringField(2, repeated=True)
    signatures = _messages.MessageField('GoogleCloudSecuritycenterV2ProcessSignature', 3, repeated=True)
    uris = _messages.StringField(4, repeated=True)
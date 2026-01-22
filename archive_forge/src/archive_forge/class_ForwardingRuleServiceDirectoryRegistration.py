from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ForwardingRuleServiceDirectoryRegistration(_messages.Message):
    """Describes the auto-registration of the forwarding rule to Service
  Directory. The region and project of the Service Directory resource
  generated from this registration will be the same as this forwarding rule.

  Fields:
    namespace: Service Directory namespace to register the forwarding rule
      under.
    service: Service Directory service to register the forwarding rule under.
    serviceDirectoryRegion: [Optional] Service Directory region to register
      this global forwarding rule under. Default to "us-central1". Only used
      for PSC for Google APIs. All PSC for Google APIs forwarding rules on the
      same network should use the same Service Directory region.
  """
    namespace = _messages.StringField(1)
    service = _messages.StringField(2)
    serviceDirectoryRegion = _messages.StringField(3)
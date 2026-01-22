from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EgressControlConfig(_messages.Message):
    """Egress control config for connector runtime. These configurations define
  the rules to identify which outbound domains/hosts needs to be whitelisted.
  It may be a static information for a particular connector version or it is
  derived from the configurations provided by the customer in Connection
  resource.

  Fields:
    backends: Static Comma separated backends which are common for all
      Connection resources. Supported formats for each backend are host:port
      or just host (host can be ip address or domain name).
    extractionRules: Extractions Rules to extract the backends from customer
      provided configuration.
  """
    backends = _messages.StringField(1)
    extractionRules = _messages.MessageField('ExtractionRules', 2)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DNSEndpointConfig(_messages.Message):
    """Describes the configuration of a DNS endpoint.

  Fields:
    enabled: Controls whether this endpoint is available.
    endpoint: Output only. The cluster's DNS endpoint configuration. A DNS
      format address. This is accessible from the public internet. Ex: uid.us-
      central1.gke.goog.
  """
    enabled = _messages.BooleanField(1)
    endpoint = _messages.StringField(2)
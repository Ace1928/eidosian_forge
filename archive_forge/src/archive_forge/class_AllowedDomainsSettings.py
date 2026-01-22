from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowedDomainsSettings(_messages.Message):
    """Configuration for IAP allowed domains. Lets you to restrict access to an
  app and allow access to only the domains that you list.

  Fields:
    domains: List of trusted domains.
    enable: Configuration for customers to opt in for the feature.
  """
    domains = _messages.StringField(1, repeated=True)
    enable = _messages.BooleanField(2)
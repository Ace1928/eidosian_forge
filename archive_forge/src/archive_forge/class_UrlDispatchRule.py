from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UrlDispatchRule(_messages.Message):
    """Rules to match an HTTP request and dispatch that request to a service.

  Fields:
    domain: Domain name to match against. The wildcard "*" is supported if
      specified before a period: "*.".Defaults to matching all domains: "*".
    path: Pathname within the host. Must start with a "/". A single "*" can be
      included at the end of the path.The sum of the lengths of the domain and
      path may not exceed 100 characters.
    service: Resource ID of a service in this application that should serve
      the matched request. The service must already exist. Example: default.
  """
    domain = _messages.StringField(1)
    path = _messages.StringField(2)
    service = _messages.StringField(3)
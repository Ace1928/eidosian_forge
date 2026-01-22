from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TcpRouteRouteMatch(_messages.Message):
    """RouteMatch defines the predicate used to match requests to a given
  action. Multiple match types are "OR"ed for evaluation. If no routeMatch
  field is specified, this rule will unconditionally match traffic.

  Fields:
    address: Required. Must be specified in the CIDR range format. A CIDR
      range consists of an IP Address and a prefix length to construct the
      subnet mask. By default, the prefix length is 32 (i.e. matches a single
      IP address). Only IPV4 addresses are supported. Examples: "10.0.0.1" -
      matches against this exact IP address. "10.0.0.0/8" - matches against
      any IP address within the 10.0.0.0 subnet and 255.255.255.0 mask.
      "0.0.0.0/0" - matches against any IP address'.
    port: Required. Specifies the destination port to match against.
  """
    address = _messages.StringField(1)
    port = _messages.StringField(2)
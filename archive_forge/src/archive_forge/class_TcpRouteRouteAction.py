from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TcpRouteRouteAction(_messages.Message):
    """The specifications for routing traffic and applying associated policies.

  Fields:
    destinations: Optional. The destination services to which traffic should
      be forwarded. At least one destination service is required. Only one of
      route destination or original destination can be set.
    idleTimeout: Optional. Specifies the idle timeout for the selected route.
      The idle timeout is defined as the period in which there are no bytes
      sent or received on either the upstream or downstream connection. If not
      set, the default idle timeout is 30 seconds. If set to 0s, the timeout
      will be disabled.
    originalDestination: Optional. If true, Router will use the destination IP
      and port of the original connection as the destination of the request.
      Default is false. Only one of route destinations or original destination
      can be set.
  """
    destinations = _messages.MessageField('TcpRouteRouteDestination', 1, repeated=True)
    idleTimeout = _messages.StringField(2)
    originalDestination = _messages.BooleanField(3)
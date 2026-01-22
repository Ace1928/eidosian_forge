from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ExpirationPolicy(_messages.Message):
    """A policy that specifies the conditions for resource expiration (i.e.,
  automatic resource deletion).

  Fields:
    ttl: Optional. Specifies the "time-to-live" duration for an associated
      resource. The resource expires if it is not active for a period of
      `ttl`. The definition of "activity" depends on the type of the
      associated resource. The minimum and maximum allowed values for `ttl`
      depend on the type of the associated resource, as well. If `ttl` is not
      set, the associated resource never expires.
  """
    ttl = _messages.StringField(1)
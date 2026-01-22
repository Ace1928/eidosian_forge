from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TunnelDestGroup(_messages.Message):
    """A TunnelDestGroup.

  Fields:
    cidrs: Unordered list. List of CIDRs that this group applies to.
    fqdns: Unordered list. List of FQDNs that this group applies to.
    name: Required. Immutable. Identifier for the TunnelDestGroup. Must be
      unique within the project and contain only lower case letters (a-z) and
      dashes (-).
  """
    cidrs = _messages.StringField(1, repeated=True)
    fqdns = _messages.StringField(2, repeated=True)
    name = _messages.StringField(3)
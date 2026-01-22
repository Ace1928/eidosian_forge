from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinkLACPStatus(_messages.Message):
    """Describing the status of a LACP link.

  Enums:
    StateValueValuesEnum: The state of a LACP link.

  Fields:
    aggregatable: A true value indicates that the participant will allow the
      link to be used as part of the aggregate. A false value indicates the
      link should be used as an individual link.
    collecting: If true, the participant is collecting incoming frames on the
      link, otherwise false
    distributing: When true, the participant is distributing outgoing frames;
      when false, distribution is disabled
    googleSystemId: System ID of the port on Google's side of the LACP
      exchange.
    neighborSystemId: System ID of the port on the neighbor's side of the LACP
      exchange.
    state: The state of a LACP link.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of a LACP link.

    Values:
      UNKNOWN: The default state indicating state is in unknown state.
      ACTIVE: The link is configured and active within the bundle.
      DETACHED: The link is not configured within the bundle, this means the
        rest of the object should be empty.
    """
        UNKNOWN = 0
        ACTIVE = 1
        DETACHED = 2
    aggregatable = _messages.BooleanField(1)
    collecting = _messages.BooleanField(2)
    distributing = _messages.BooleanField(3)
    googleSystemId = _messages.StringField(4)
    neighborSystemId = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
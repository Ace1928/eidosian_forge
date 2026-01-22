from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FleetConstraintTemplate(_messages.Message):
    """FleetConstraintTemplate contains aggregate status for a single
  constraint template. The aggregation is across all member clusters in a
  fleet.

  Fields:
    numConstraints: The number of unique constraints using this constraint
      template. This is included so avoid clients don't have to make serial
      round trips.
    numMemberships: The number of member clusters on which this constraint
      template was found.
    ref: The constraint template this data refers to.
  """
    numConstraints = _messages.IntegerField(1)
    numMemberships = _messages.IntegerField(2)
    ref = _messages.MessageField('ConstraintTemplateRef', 3)
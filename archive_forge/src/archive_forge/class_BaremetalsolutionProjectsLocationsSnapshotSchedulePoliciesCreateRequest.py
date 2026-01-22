from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesCreateRequest(_messages.Message):
    """A
  BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesCreateRequest
  object.

  Fields:
    parent: Required. The parent project and location containing the
      SnapshotSchedulePolicy.
    snapshotSchedulePolicy: A SnapshotSchedulePolicy resource to be passed as
      the request body.
    snapshotSchedulePolicyId: Required. Snapshot policy ID
  """
    parent = _messages.StringField(1, required=True)
    snapshotSchedulePolicy = _messages.MessageField('SnapshotSchedulePolicy', 2)
    snapshotSchedulePolicyId = _messages.StringField(3)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DemoteContext(_messages.Message):
    """This context is used to demote an existing standalone instance to be a
  Cloud SQL read replica for an external database server.

  Fields:
    kind: This is always `sql#demoteContext`.
    sourceRepresentativeInstanceName: Required. The name of the instance which
      acts as an on-premises primary instance in the replication setup.
  """
    kind = _messages.StringField(1)
    sourceRepresentativeInstanceName = _messages.StringField(2)
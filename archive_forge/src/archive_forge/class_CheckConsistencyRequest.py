from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckConsistencyRequest(_messages.Message):
    """Request message for
  google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency

  Fields:
    consistencyToken: Required. The token created using
      GenerateConsistencyToken for the Table.
    dataBoostReadLocalWrites: Checks that reads using an app profile with
      `DataBoostIsolationReadOnly` can see all writes committed before the
      token was created, but only if the read and write target the same
      cluster.
    standardReadRemoteWrites: Checks that reads using an app profile with
      `StandardIsolation` can see all writes committed before the token was
      created, even if the read and write target different clusters.
  """
    consistencyToken = _messages.StringField(1)
    dataBoostReadLocalWrites = _messages.MessageField('DataBoostReadLocalWrites', 2)
    standardReadRemoteWrites = _messages.MessageField('StandardReadRemoteWrites', 3)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckConsistencyResponse(_messages.Message):
    """Response message for
  google.bigtable.admin.v2.BigtableTableAdmin.CheckConsistency

  Fields:
    consistent: True only if the token is consistent. A token is consistent if
      replication has caught up with the restrictions specified in the
      request.
  """
    consistent = _messages.BooleanField(1)
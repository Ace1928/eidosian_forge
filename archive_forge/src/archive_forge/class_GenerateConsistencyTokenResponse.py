from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateConsistencyTokenResponse(_messages.Message):
    """Response message for
  google.bigtable.admin.v2.BigtableTableAdmin.GenerateConsistencyToken

  Fields:
    consistencyToken: The generated consistency token.
  """
    consistencyToken = _messages.StringField(1)
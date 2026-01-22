from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollbackApiSpecRequest(_messages.Message):
    """Request message for RollbackApiSpec.

  Fields:
    revisionId: Required. The revision ID to roll back to. It must be a
      revision of the same spec. Example: `c7cfa2a8`
  """
    revisionId = _messages.StringField(1)
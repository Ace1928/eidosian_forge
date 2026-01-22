from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UndeleteServiceAccountResponse(_messages.Message):
    """A UndeleteServiceAccountResponse object.

  Fields:
    restoredAccount: Metadata for the restored service account.
  """
    restoredAccount = _messages.MessageField('ServiceAccount', 1)
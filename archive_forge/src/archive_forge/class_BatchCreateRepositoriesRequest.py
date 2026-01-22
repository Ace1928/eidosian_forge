from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateRepositoriesRequest(_messages.Message):
    """Message for creating repositoritories in batch.

  Fields:
    requests: Required. The request messages specifying the repositories to
      create.
  """
    requests = _messages.MessageField('CreateRepositoryRequest', 1, repeated=True)
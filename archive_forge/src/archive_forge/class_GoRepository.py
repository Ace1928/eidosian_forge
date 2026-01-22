from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoRepository(_messages.Message):
    """Configuration for a Go remote repository.

  Fields:
    customRepository: One of the publicly available Go repositories.
  """
    customRepository = _messages.MessageField('CustomRepository', 1)
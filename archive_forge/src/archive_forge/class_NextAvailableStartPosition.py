from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NextAvailableStartPosition(_messages.Message):
    """CDC strategy to resume replication from the next available position in
  the source.
  """
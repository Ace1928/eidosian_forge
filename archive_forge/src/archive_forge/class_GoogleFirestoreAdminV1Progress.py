from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1Progress(_messages.Message):
    """Describes the progress of the operation. Unit of work is generic and
  must be interpreted based on where Progress is used.

  Fields:
    completedWork: The amount of work completed.
    estimatedWork: The amount of work estimated.
  """
    completedWork = _messages.IntegerField(1)
    estimatedWork = _messages.IntegerField(2)
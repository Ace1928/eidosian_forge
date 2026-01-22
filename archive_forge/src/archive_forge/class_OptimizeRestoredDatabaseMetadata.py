from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OptimizeRestoredDatabaseMetadata(_messages.Message):
    """Metadata type for the long-running operation used to track the progress
  of optimizations performed on a newly restored database. This long-running
  operation is automatically created by the system after the successful
  completion of a database restore, and cannot be cancelled.

  Fields:
    name: Name of the restored database being optimized.
    progress: The progress of the post-restore optimizations.
  """
    name = _messages.StringField(1)
    progress = _messages.MessageField('OperationProgress', 2)
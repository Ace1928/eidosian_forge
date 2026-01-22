from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeSnapshotsGetRequest(_messages.Message):
    """A ComputeSnapshotsGetRequest object.

  Fields:
    project: Project ID for this request.
    snapshot: Name of the Snapshot resource to return.
  """
    project = _messages.StringField(1, required=True)
    snapshot = _messages.StringField(2, required=True)
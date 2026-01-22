from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobStatistics4(_messages.Message):
    """A JobStatistics4 object.

  Fields:
    destinationUriFileCounts: [Output-only] Number of files per destination
      URI or URI pattern specified in the extract configuration. These values
      will be in the same order as the URIs specified in the 'destinationUris'
      field.
  """
    destinationUriFileCounts = _messages.IntegerField(1, repeated=True)
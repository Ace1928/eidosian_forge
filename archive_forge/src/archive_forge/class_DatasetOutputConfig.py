from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatasetOutputConfig(_messages.Message):
    """Output configuration for datasets.

  Fields:
    gcsDestination: Google Cloud Storage destination to write the output.
  """
    gcsDestination = _messages.MessageField('GcsOutputDestination', 1)
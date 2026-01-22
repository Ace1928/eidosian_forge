from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportProductSetsInputConfig(_messages.Message):
    """The input content for the `ImportProductSets` method.

  Fields:
    gcsSource: The Google Cloud Storage location for a csv file which
      preserves a list of ImportProductSetRequests in each line.
  """
    gcsSource = _messages.MessageField('ImportProductSetsGcsSource', 1)
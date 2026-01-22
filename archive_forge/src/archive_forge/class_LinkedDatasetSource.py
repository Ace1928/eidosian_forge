from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinkedDatasetSource(_messages.Message):
    """A dataset source type which refers to another BigQuery dataset.

  Fields:
    sourceDataset: The source dataset reference contains project numbers and
      not project ids.
  """
    sourceDataset = _messages.MessageField('DatasetReference', 1)
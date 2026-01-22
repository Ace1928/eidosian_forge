from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceHierarchyDatasets(_messages.Message):
    """Destination datasets are created so that hierarchy of the destination
  data objects matches the source hierarchy.

  Fields:
    datasetTemplate: The dataset template to use for dynamic dataset creation.
  """
    datasetTemplate = _messages.MessageField('DatasetTemplate', 1)
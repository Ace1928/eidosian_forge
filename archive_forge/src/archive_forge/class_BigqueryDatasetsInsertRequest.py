from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryDatasetsInsertRequest(_messages.Message):
    """A BigqueryDatasetsInsertRequest object.

  Fields:
    dataset: A Dataset resource to be passed as the request body.
    projectId: Project ID of the new dataset
  """
    dataset = _messages.MessageField('Dataset', 1)
    projectId = _messages.StringField(2, required=True)
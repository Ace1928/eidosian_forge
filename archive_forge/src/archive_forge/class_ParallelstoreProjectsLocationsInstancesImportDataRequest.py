from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParallelstoreProjectsLocationsInstancesImportDataRequest(_messages.Message):
    """A ParallelstoreProjectsLocationsInstancesImportDataRequest object.

  Fields:
    importDataRequest: A ImportDataRequest resource to be passed as the
      request body.
    name: Required. Name of the resource.
  """
    importDataRequest = _messages.MessageField('ImportDataRequest', 1)
    name = _messages.StringField(2, required=True)
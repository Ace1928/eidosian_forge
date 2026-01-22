from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookerProjectsLocationsInstancesImportRequest(_messages.Message):
    """A LookerProjectsLocationsInstancesImportRequest object.

  Fields:
    importInstanceRequest: A ImportInstanceRequest resource to be passed as
      the request body.
    name: Required. Format:
      `projects/{project}/locations/{location}/instances/{instance}`.
  """
    importInstanceRequest = _messages.MessageField('ImportInstanceRequest', 1)
    name = _messages.StringField(2, required=True)
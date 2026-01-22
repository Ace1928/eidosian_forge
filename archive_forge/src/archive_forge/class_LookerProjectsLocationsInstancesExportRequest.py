from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookerProjectsLocationsInstancesExportRequest(_messages.Message):
    """A LookerProjectsLocationsInstancesExportRequest object.

  Fields:
    exportInstanceRequest: A ExportInstanceRequest resource to be passed as
      the request body.
    name: Required. Format:
      `projects/{project}/locations/{location}/instances/{instance}`.
  """
    exportInstanceRequest = _messages.MessageField('ExportInstanceRequest', 1)
    name = _messages.StringField(2, required=True)
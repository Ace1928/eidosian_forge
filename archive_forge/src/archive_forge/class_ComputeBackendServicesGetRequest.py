from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeBackendServicesGetRequest(_messages.Message):
    """A ComputeBackendServicesGetRequest object.

  Fields:
    backendService: Name of the BackendService resource to return.
    project: Project ID for this request.
  """
    backendService = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
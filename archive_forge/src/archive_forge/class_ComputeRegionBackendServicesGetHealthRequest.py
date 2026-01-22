from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRegionBackendServicesGetHealthRequest(_messages.Message):
    """A ComputeRegionBackendServicesGetHealthRequest object.

  Fields:
    backendService: Name of the BackendService resource for which to get
      health.
    project: A string attribute.
    region: Name of the region scoping this request.
    resourceGroupReference: A ResourceGroupReference resource to be passed as
      the request body.
  """
    backendService = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)
    resourceGroupReference = _messages.MessageField('ResourceGroupReference', 4)
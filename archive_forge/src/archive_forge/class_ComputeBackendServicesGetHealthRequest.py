from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeBackendServicesGetHealthRequest(_messages.Message):
    """A ComputeBackendServicesGetHealthRequest object.

  Fields:
    backendService: Name of the BackendService resource to which the queried
      instance belongs.
    project: A string attribute.
    resourceGroupReference: A ResourceGroupReference resource to be passed as
      the request body.
  """
    backendService = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    resourceGroupReference = _messages.MessageField('ResourceGroupReference', 3)
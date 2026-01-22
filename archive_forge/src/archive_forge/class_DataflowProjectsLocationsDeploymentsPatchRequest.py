from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsDeploymentsPatchRequest(_messages.Message):
    """A DataflowProjectsLocationsDeploymentsPatchRequest object.

  Fields:
    allowMissing: If set to true, and the `Deployment` is not found, a new
      `Deployment` will be created.
    deployment: A Deployment resource to be passed as the request body.
    name: Required. The name of the `Deployment`. Format:
      projects/{project}/locations/{location}/deployments/{deployment_id}
    requestId: A unique identifier for this request. Restricted to 36 ASCII
      characters. A random UUID is recommended. This request is only
      idempotent if a `request_id` is provided.
    updateMask: The list of fields to update.
    validateOnly: Validate an intended change to see what the result will be
      before actually making the change.
  """
    allowMissing = _messages.BooleanField(1)
    deployment = _messages.MessageField('Deployment', 2)
    name = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)
    updateMask = _messages.StringField(5)
    validateOnly = _messages.BooleanField(6)
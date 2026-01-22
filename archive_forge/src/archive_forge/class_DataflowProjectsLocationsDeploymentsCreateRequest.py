from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsDeploymentsCreateRequest(_messages.Message):
    """A DataflowProjectsLocationsDeploymentsCreateRequest object.

  Fields:
    deployment: A Deployment resource to be passed as the request body.
    deploymentId: The ID to use for the Cloud Dataflow job deployment, which
      will become the final component of the deployment's resource name. This
      value should be 4-63 characters, and valid characters are /a-z-/.
    parent: Required. The parent resource where this deployment will be
      created. Format: projects/{project}/locations/{location}
    requestId: A unique identifier for this request. Restricted to 36 ASCII
      characters. A random UUID is recommended. This request is only
      idempotent if a `request_id` is provided.
    validateOnly: Validate an intended change to see what the result will be
      before actually making the change.
  """
    deployment = _messages.MessageField('Deployment', 1)
    deploymentId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)
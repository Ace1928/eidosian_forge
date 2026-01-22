from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsDeploymentsDeleteRequest(_messages.Message):
    """A DataflowProjectsLocationsDeploymentsDeleteRequest object.

  Fields:
    allowMissing: If set to true, and the `Deployment` is not found, the
      request will succeed but no action will be taken on the server.
    etag: The current etag of the Deployment. Checksum computed by the server.
      Ensure that the client has an up-to-date value before proceeding. If an
      etag is provided and does not match the current etag of the
      `Deployment`, request will be blocked and an ABORTED error will be
      returned.
    name: Required. The name of the `Deployment`. Format:
      projects/{project}/locations/{location}/deployments/{deployment_id}
    validateOnly: Validate an intended change to see what the result will be
      before actually making the change.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)
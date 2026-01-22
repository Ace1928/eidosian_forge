from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GsuiteaddonsProjectsDeploymentsDeleteRequest(_messages.Message):
    """A GsuiteaddonsProjectsDeploymentsDeleteRequest object.

  Fields:
    etag: The etag of the deployment to delete. If this is provided, it must
      match the server's etag.
    name: Required. The full resource name of the deployment to delete.
      Example: `projects/my_project/deployments/my_deployment`.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
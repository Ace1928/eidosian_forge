from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGsuiteaddonsV1InstallStatus(_messages.Message):
    """Install status of a test deployment.

  Fields:
    installed: True if the deployment is installed for the user.
    name: The canonical full resource name of the deployment install status.
      Example: `projects/123/deployments/my_deployment/installStatus`.
  """
    installed = _messages.BooleanField(1)
    name = _messages.StringField(2)
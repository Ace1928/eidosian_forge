from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalNodesNodesDeploymentsCreateRequest(_messages.Message):
    """A SasportalNodesNodesDeploymentsCreateRequest object.

  Fields:
    parent: Required. The parent resource name where the deployment is to be
      created.
    sasPortalDeployment: A SasPortalDeployment resource to be passed as the
      request body.
  """
    parent = _messages.StringField(1, required=True)
    sasPortalDeployment = _messages.MessageField('SasPortalDeployment', 2)
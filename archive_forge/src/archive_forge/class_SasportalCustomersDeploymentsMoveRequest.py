from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalCustomersDeploymentsMoveRequest(_messages.Message):
    """A SasportalCustomersDeploymentsMoveRequest object.

  Fields:
    name: Required. The name of the deployment to move.
    sasPortalMoveDeploymentRequest: A SasPortalMoveDeploymentRequest resource
      to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    sasPortalMoveDeploymentRequest = _messages.MessageField('SasPortalMoveDeploymentRequest', 2)
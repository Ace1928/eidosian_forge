from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollbackDeploymentRequest(_messages.Message):
    """Request object for `RollbackDeployment`.

  Fields:
    revisionId: Required. The revision id of deployment to roll back to.
  """
    revisionId = _messages.StringField(1)
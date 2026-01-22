from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentNote(_messages.Message):
    """An artifact that can be deployed in some runtime.

  Fields:
    resourceUri: Required. Resource URI for the artifact being deployed.
  """
    resourceUri = _messages.StringField(1, repeated=True)
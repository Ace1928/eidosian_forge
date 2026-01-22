from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerCompositeTypesInsertRequest(_messages.Message):
    """A DeploymentmanagerCompositeTypesInsertRequest object.

  Fields:
    compositeType: A CompositeType resource to be passed as the request body.
    project: The project ID for this request.
  """
    compositeType = _messages.MessageField('CompositeType', 1)
    project = _messages.StringField(2, required=True)
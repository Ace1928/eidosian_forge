from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrafeasV1beta1DeploymentDetails(_messages.Message):
    """Details of a deployment occurrence.

  Fields:
    deployment: Required. Deployment history for the resource.
  """
    deployment = _messages.MessageField('Deployment', 1)
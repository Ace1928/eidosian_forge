from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListDeploymentsResponse(_messages.Message):
    """A GoogleCloudApigeeV1ListDeploymentsResponse object.

  Fields:
    deployments: List of deployments.
  """
    deployments = _messages.MessageField('GoogleCloudApigeeV1Deployment', 1, repeated=True)
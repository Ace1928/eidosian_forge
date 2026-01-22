from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalListGcpProjectDeploymentsResponse(_messages.Message):
    """Response for [ListGcpProjectDeployments].

  Fields:
    deployments: Optional. Deployments associated with the GCP project
  """
    deployments = _messages.MessageField('SasPortalGcpProjectDeployment', 1, repeated=True)
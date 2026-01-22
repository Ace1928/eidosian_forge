from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2LoadBalancer(_messages.Message):
    """Contains information related to the load balancer associated with the
  finding.

  Fields:
    name: The name of the load balancer associated with the finding.
  """
    name = _messages.StringField(1)
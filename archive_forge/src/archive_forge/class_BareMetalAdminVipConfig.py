from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminVipConfig(_messages.Message):
    """BareMetalAdminVipConfig for bare metal load balancer configurations.

  Fields:
    controlPlaneVip: The VIP which you previously set aside for the Kubernetes
      API of this bare metal admin cluster.
  """
    controlPlaneVip = _messages.StringField(1)
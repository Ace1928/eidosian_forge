from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneMaintenanceConfig(_messages.Message):
    """Specifies configurations to put bare metal nodes in and out of
  maintenance.

  Fields:
    maintenanceAddressCidrBlocks: Required. All IPv4 address from these ranges
      will be placed into maintenance mode. Nodes in maintenance mode will be
      cordoned and drained. When both of these are true, the
      "baremetal.cluster.gke.io/maintenance" annotation will be set on the
      node resource.
  """
    maintenanceAddressCidrBlocks = _messages.StringField(1, repeated=True)
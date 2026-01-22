from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1UpdateSettings(_messages.Message):
    """UpdateSettings control the level of parallelism and the level of
  disruption caused during the update of a node pool. These settings are
  applicable when the node pool update requires replacing the existing node
  pool nodes with the updated ones. UpdateSettings are optional. When
  UpdateSettings are not specified during the node pool creation, a default is
  chosen based on the parent cluster's version. For clusters with minor
  version 1.27 and later, a default surge_settings configuration with
  max_surge = 1 and max_unavailable = 0 is used. For clusters with older
  versions, node pool updates use the traditional rolling update mechanism of
  updating one node at a time in a "terminate before create" fashion and
  update_settings is not applicable. Set the surge_settings parameter to use
  the Surge Update mechanism for the rolling update of node pool nodes. 1.
  max_surge controls the number of additional nodes that can be created beyond
  the current size of the node pool temporarily for the time of the update to
  increase the number of available nodes. 2. max_unavailable controls the
  number of nodes that can be simultaneously unavailable during the update. 3.
  (max_surge + max_unavailable) determines the level of parallelism (i.e., the
  number of nodes being updated at the same time).

  Fields:
    surgeSettings: Optional. Settings for surge update.
  """
    surgeSettings = _messages.MessageField('GoogleCloudGkemulticloudV1SurgeSettings', 1)
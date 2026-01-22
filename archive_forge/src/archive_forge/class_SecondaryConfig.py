from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecondaryConfig(_messages.Message):
    """Configuration information for the secondary cluster. This should be set
  if and only if the cluster is of type SECONDARY.

  Fields:
    primaryClusterName: The name of the primary cluster name with the format:
      * projects/{project}/locations/{region}/clusters/{cluster_id}
  """
    primaryClusterName = _messages.StringField(1)
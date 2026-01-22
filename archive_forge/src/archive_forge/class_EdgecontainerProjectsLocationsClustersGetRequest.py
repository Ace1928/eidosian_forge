from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgecontainerProjectsLocationsClustersGetRequest(_messages.Message):
    """A EdgecontainerProjectsLocationsClustersGetRequest object.

  Fields:
    name: Required. The resource name of the cluster.
  """
    name = _messages.StringField(1, required=True)
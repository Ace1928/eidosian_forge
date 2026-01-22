from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgeCluster(_messages.Message):
    """EdgeCluster contains information specific to Google Edge Clusters.

  Fields:
    resourceLink: Immutable. Self-link of the Google Cloud resource for the
      Edge Cluster. For example: //edgecontainer.googleapis.com/projects/my-
      project/locations/us-west1-a/clusters/my-cluster
  """
    resourceLink = _messages.StringField(1)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplianceCluster(_messages.Message):
    """ApplianceCluster contains information specific to GDC Edge Appliance
  Clusters.

  Fields:
    resourceLink: Immutable. Self-link of the Google Cloud resource for the
      Appliance Cluster. For example:
      //transferappliance.googleapis.com/projects/my-project/locations/us-
      west1-a/appliances/my-appliance
  """
    resourceLink = _messages.StringField(1)
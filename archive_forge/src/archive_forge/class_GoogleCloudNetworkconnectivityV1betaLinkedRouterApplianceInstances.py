from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaLinkedRouterApplianceInstances(_messages.Message):
    """A collection of router appliance instances. If you configure multiple
  router appliance instances to receive data from the same set of sites
  outside of Google Cloud, we recommend that you associate those instances
  with the same spoke.

  Fields:
    instances: The list of router appliance instances.
    siteToSiteDataTransfer: A value that controls whether site-to-site data
      transfer is enabled for these resources. Data transfer is available only
      in [supported locations](https://cloud.google.com/network-
      connectivity/docs/network-connectivity-center/concepts/locations).
    vpcNetwork: Output only. The VPC network where these router appliance
      instances are located.
  """
    instances = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaRouterApplianceInstance', 1, repeated=True)
    siteToSiteDataTransfer = _messages.BooleanField(2)
    vpcNetwork = _messages.StringField(3)
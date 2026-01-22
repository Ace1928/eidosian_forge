from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoprovisioningNodePoolDefaults(_messages.Message):
    """AutoprovisioningNodePoolDefaults contains defaults for a node pool
  created by NAP.

  Fields:
    bootDiskKmsKey: The Customer Managed Encryption Key used to encrypt the
      boot disk attached to each node in the node pool. This should be of the
      form projects/[KEY_PROJECT_ID]/locations/[LOCATION]/keyRings/[RING_NAME]
      /cryptoKeys/[KEY_NAME]. For more information about protecting resources
      with Cloud KMS Keys please see:
      https://cloud.google.com/compute/docs/disks/customer-managed-encryption
    diskSizeGb: Size of the disk attached to each node, specified in GB. The
      smallest allowed disk size is 10GB. If unspecified, the default disk
      size is 100GB.
    diskType: Type of the disk attached to each node (e.g. 'pd-standard', 'pd-
      ssd' or 'pd-balanced') If unspecified, the default disk type is 'pd-
      standard'
    imageType: The image type to use for NAP created node. Please see
      https://cloud.google.com/kubernetes-engine/docs/concepts/node-images for
      available image types.
    insecureKubeletReadonlyPortEnabled: Enable or disable Kubelet read only
      port.
    management: Specifies the node management options for NAP created node-
      pools.
    minCpuPlatform: Deprecated. Minimum CPU platform to be used for NAP
      created node pools. The instance may be scheduled on the specified or
      newer CPU platform. Applicable values are the friendly names of CPU
      platforms, such as minCpuPlatform: Intel Haswell or minCpuPlatform:
      Intel Sandy Bridge. For more information, read [how to specify min CPU
      platform](https://cloud.google.com/compute/docs/instances/specify-min-
      cpu-platform). This field is deprecated, min_cpu_platform should be
      specified using `cloud.google.com/requested-min-cpu-platform` label
      selector on the pod. To unset the min cpu platform field pass
      "automatic" as field value.
    oauthScopes: Scopes that are used by NAP when creating node pools.
    serviceAccount: The Google Cloud Platform Service Account to be used by
      the node VMs.
    shieldedInstanceConfig: Shielded Instance options.
    upgradeSettings: Specifies the upgrade settings for NAP created node pools
  """
    bootDiskKmsKey = _messages.StringField(1)
    diskSizeGb = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    diskType = _messages.StringField(3)
    imageType = _messages.StringField(4)
    insecureKubeletReadonlyPortEnabled = _messages.BooleanField(5)
    management = _messages.MessageField('NodeManagement', 6)
    minCpuPlatform = _messages.StringField(7)
    oauthScopes = _messages.StringField(8, repeated=True)
    serviceAccount = _messages.StringField(9)
    shieldedInstanceConfig = _messages.MessageField('ShieldedInstanceConfig', 10)
    upgradeSettings = _messages.MessageField('UpgradeSettings', 11)
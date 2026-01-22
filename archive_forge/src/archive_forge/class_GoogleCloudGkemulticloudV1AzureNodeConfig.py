from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureNodeConfig(_messages.Message):
    """Parameters that describe the configuration of all node machines on a
  given node pool.

  Messages:
    LabelsValue: Optional. The initial labels assigned to nodes of this node
      pool. An object containing a list of "key": value pairs. Example: {
      "name": "wrench", "mass": "1.3kg", "count": "3" }.
    TagsValue: Optional. A set of tags to apply to all underlying Azure
      resources for this node pool. This currently only includes Virtual
      Machine Scale Sets. Specify at most 50 pairs containing alphanumerics,
      spaces, and symbols (.+-=_:@/). Keys can be up to 127 Unicode
      characters. Values can be up to 255 Unicode characters.

  Fields:
    configEncryption: Optional. Configuration related to vm config encryption.
    imageType: Optional. The OS image type to use on node pool instances. Can
      be unspecified, or have a value of `ubuntu`. When unspecified, it
      defaults to `ubuntu`.
    labels: Optional. The initial labels assigned to nodes of this node pool.
      An object containing a list of "key": value pairs. Example: { "name":
      "wrench", "mass": "1.3kg", "count": "3" }.
    proxyConfig: Optional. Proxy configuration for outbound HTTP(S) traffic.
    rootVolume: Optional. Configuration related to the root volume provisioned
      for each node pool machine. When unspecified, it defaults to a 32-GiB
      Azure Disk.
    sshConfig: Required. SSH configuration for how to access the node pool
      machines.
    tags: Optional. A set of tags to apply to all underlying Azure resources
      for this node pool. This currently only includes Virtual Machine Scale
      Sets. Specify at most 50 pairs containing alphanumerics, spaces, and
      symbols (.+-=_:@/). Keys can be up to 127 Unicode characters. Values can
      be up to 255 Unicode characters.
    taints: Optional. The initial taints assigned to nodes of this node pool.
    vmSize: Optional. The Azure VM size name. Example: `Standard_DS2_v2`. See
      [Supported VM sizes](/anthos/clusters/docs/azure/reference/supported-
      vms) for options. When unspecified, it defaults to `Standard_DS2_v2`.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The initial labels assigned to nodes of this node pool. An
    object containing a list of "key": value pairs. Example: { "name":
    "wrench", "mass": "1.3kg", "count": "3" }.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TagsValue(_messages.Message):
        """Optional. A set of tags to apply to all underlying Azure resources for
    this node pool. This currently only includes Virtual Machine Scale Sets.
    Specify at most 50 pairs containing alphanumerics, spaces, and symbols
    (.+-=_:@/). Keys can be up to 127 Unicode characters. Values can be up to
    255 Unicode characters.

    Messages:
      AdditionalProperty: An additional property for a TagsValue object.

    Fields:
      additionalProperties: Additional properties of type TagsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    configEncryption = _messages.MessageField('GoogleCloudGkemulticloudV1AzureConfigEncryption', 1)
    imageType = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    proxyConfig = _messages.MessageField('GoogleCloudGkemulticloudV1AzureProxyConfig', 4)
    rootVolume = _messages.MessageField('GoogleCloudGkemulticloudV1AzureDiskTemplate', 5)
    sshConfig = _messages.MessageField('GoogleCloudGkemulticloudV1AzureSshConfig', 6)
    tags = _messages.MessageField('TagsValue', 7)
    taints = _messages.MessageField('GoogleCloudGkemulticloudV1NodeTaint', 8, repeated=True)
    vmSize = _messages.StringField(9)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1IndexEndpoint(_messages.Message):
    """Indexes are deployed into it. An IndexEndpoint can have multiple
  DeployedIndexes.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize your
      IndexEndpoints. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.

  Fields:
    createTime: Output only. Timestamp when this IndexEndpoint was created.
    deployedIndexes: Output only. The indexes deployed in this endpoint.
    description: The description of the IndexEndpoint.
    displayName: Required. The display name of the IndexEndpoint. The name can
      be up to 128 characters long and can consist of any UTF-8 characters.
    enablePrivateServiceConnect: Optional. Deprecated: If true, expose the
      IndexEndpoint via private service connect. Only one of the fields,
      network or enable_private_service_connect, can be set.
    encryptionSpec: Immutable. Customer-managed encryption key spec for an
      IndexEndpoint. If set, this IndexEndpoint and all sub-resources of this
      IndexEndpoint will be secured by this key.
    etag: Used to perform consistent read-modify-write updates. If not set, a
      blind "overwrite" update happens.
    labels: The labels with user-defined metadata to organize your
      IndexEndpoints. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.
    name: Output only. The resource name of the IndexEndpoint.
    network: Optional. The full name of the Google Compute Engine
      [network](https://cloud.google.com/compute/docs/networks-and-
      firewalls#networks) to which the IndexEndpoint should be peered. Private
      services access must already be configured for the network. If left
      unspecified, the Endpoint is not peered with any network. network and
      private_service_connect_config are mutually exclusive. [Format](https://
      cloud.google.com/compute/docs/reference/rest/v1/networks/insert):
      `projects/{project}/global/networks/{network}`. Where {project} is a
      project number, as in '12345', and {network} is network name.
    privateServiceConnectConfig: Optional. Configuration for private service
      connect. network and private_service_connect_config are mutually
      exclusive.
    publicEndpointDomainName: Output only. If public_endpoint_enabled is true,
      this field will be populated with the domain name to use for this index
      endpoint.
    publicEndpointEnabled: Optional. If true, the deployed index will be
      accessible through public endpoint.
    updateTime: Output only. Timestamp when this IndexEndpoint was last
      updated. This timestamp is not updated when the endpoint's
      DeployedIndexes are updated, e.g. due to updates of the original Indexes
      they are the deployments of.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize your IndexEndpoints.
    Label keys and values can be no longer than 64 characters (Unicode
    codepoints), can only contain lowercase letters, numeric characters,
    underscores and dashes. International characters are allowed. See
    https://goo.gl/xmQnxf for more information and examples of labels.

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
    createTime = _messages.StringField(1)
    deployedIndexes = _messages.MessageField('GoogleCloudAiplatformV1DeployedIndex', 2, repeated=True)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    enablePrivateServiceConnect = _messages.BooleanField(5)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1EncryptionSpec', 6)
    etag = _messages.StringField(7)
    labels = _messages.MessageField('LabelsValue', 8)
    name = _messages.StringField(9)
    network = _messages.StringField(10)
    privateServiceConnectConfig = _messages.MessageField('GoogleCloudAiplatformV1PrivateServiceConnectConfig', 11)
    publicEndpointDomainName = _messages.StringField(12)
    publicEndpointEnabled = _messages.BooleanField(13)
    updateTime = _messages.StringField(14)
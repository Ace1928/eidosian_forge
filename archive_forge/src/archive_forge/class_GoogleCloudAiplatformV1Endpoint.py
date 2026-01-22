from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Endpoint(_messages.Message):
    """Models are deployed into it, and afterwards Endpoint is called to obtain
  predictions and explanations.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize your
      Endpoints. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.
    TrafficSplitValue: A map from a DeployedModel's ID to the percentage of
      this Endpoint's traffic that should be forwarded to that DeployedModel.
      If a DeployedModel's ID is not listed in this map, then it receives no
      traffic. The traffic percentage values must add up to 100, or map must
      be empty if the Endpoint is to not accept any traffic at a moment.

  Fields:
    createTime: Output only. Timestamp when this Endpoint was created.
    deployedModels: Output only. The models deployed in this Endpoint. To add
      or remove DeployedModels use EndpointService.DeployModel and
      EndpointService.UndeployModel respectively.
    description: The description of the Endpoint.
    displayName: Required. The display name of the Endpoint. The name can be
      up to 128 characters long and can consist of any UTF-8 characters.
    enablePrivateServiceConnect: Deprecated: If true, expose the Endpoint via
      private service connect. Only one of the fields, network or
      enable_private_service_connect, can be set.
    encryptionSpec: Customer-managed encryption key spec for an Endpoint. If
      set, this Endpoint and all sub-resources of this Endpoint will be
      secured by this key.
    etag: Used to perform consistent read-modify-write updates. If not set, a
      blind "overwrite" update happens.
    labels: The labels with user-defined metadata to organize your Endpoints.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. See
      https://goo.gl/xmQnxf for more information and examples of labels.
    modelDeploymentMonitoringJob: Output only. Resource name of the Model
      Monitoring job associated with this Endpoint if monitoring is enabled by
      JobService.CreateModelDeploymentMonitoringJob. Format: `projects/{projec
      t}/locations/{location}/modelDeploymentMonitoringJobs/{model_deployment_
      monitoring_job}`
    name: Output only. The resource name of the Endpoint.
    network: Optional. The full name of the Google Compute Engine
      [network](https://cloud.google.com//compute/docs/networks-and-
      firewalls#networks) to which the Endpoint should be peered. Private
      services access must already be configured for the network. If left
      unspecified, the Endpoint is not peered with any network. Only one of
      the fields, network or enable_private_service_connect, can be set. [Form
      at](https://cloud.google.com/compute/docs/reference/rest/v1/networks/ins
      ert): `projects/{project}/global/networks/{network}`. Where `{project}`
      is a project number, as in `12345`, and `{network}` is network name.
    predictRequestResponseLoggingConfig: Configures the request-response
      logging for online prediction.
    trafficSplit: A map from a DeployedModel's ID to the percentage of this
      Endpoint's traffic that should be forwarded to that DeployedModel. If a
      DeployedModel's ID is not listed in this map, then it receives no
      traffic. The traffic percentage values must add up to 100, or map must
      be empty if the Endpoint is to not accept any traffic at a moment.
    updateTime: Output only. Timestamp when this Endpoint was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize your Endpoints.
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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TrafficSplitValue(_messages.Message):
        """A map from a DeployedModel's ID to the percentage of this Endpoint's
    traffic that should be forwarded to that DeployedModel. If a
    DeployedModel's ID is not listed in this map, then it receives no traffic.
    The traffic percentage values must add up to 100, or map must be empty if
    the Endpoint is to not accept any traffic at a moment.

    Messages:
      AdditionalProperty: An additional property for a TrafficSplitValue
        object.

    Fields:
      additionalProperties: Additional properties of type TrafficSplitValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TrafficSplitValue object.

      Fields:
        key: Name of the additional property.
        value: A integer attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2, variant=_messages.Variant.INT32)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    deployedModels = _messages.MessageField('GoogleCloudAiplatformV1DeployedModel', 2, repeated=True)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    enablePrivateServiceConnect = _messages.BooleanField(5)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1EncryptionSpec', 6)
    etag = _messages.StringField(7)
    labels = _messages.MessageField('LabelsValue', 8)
    modelDeploymentMonitoringJob = _messages.StringField(9)
    name = _messages.StringField(10)
    network = _messages.StringField(11)
    predictRequestResponseLoggingConfig = _messages.MessageField('GoogleCloudAiplatformV1PredictRequestResponseLoggingConfig', 12)
    trafficSplit = _messages.MessageField('TrafficSplitValue', 13)
    updateTime = _messages.StringField(14)
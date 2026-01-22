from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudFunction(_messages.Message):
    """Describes a Cloud Function that contains user computation executed in
  response to an event. It encapsulate function and triggers configurations.

  Enums:
    DockerRegistryValueValuesEnum: Docker Registry to use for this deployment.
      If unspecified, it defaults to `ARTIFACT_REGISTRY`. If
      `docker_repository` field is specified, this field should either be left
      unspecified or set to `ARTIFACT_REGISTRY`.
    IngressSettingsValueValuesEnum: The ingress settings for the function,
      controlling what traffic can reach it.
    StatusValueValuesEnum: Output only. Status of the function deployment.
    VpcConnectorEgressSettingsValueValuesEnum: The egress settings for the
      connector, controlling what traffic is diverted through it.

  Messages:
    BuildEnvironmentVariablesValue: Build environment variables that shall be
      available during build time.
    EnvironmentVariablesValue: Environment variables that shall be available
      during function execution.
    LabelsValue: Labels associated with this Cloud Function.

  Fields:
    automaticUpdatePolicy: A AutomaticUpdatePolicy attribute.
    availableMemoryMb: The amount of memory in MB available for a function.
      Defaults to 256MB.
    buildDockerfile: Local path to the dockerfile for customizing the base
      image for the builder, located within the source folder.
    buildEnvironmentVariables: Build environment variables that shall be
      available during build time.
    buildId: Output only. The Cloud Build ID of the latest successful
      deployment of the function.
    buildName: Output only. The Cloud Build Name of the function deployment.
      `projects//locations//builds/`.
    buildServiceAccount: Optional. A service account the user provides for use
      with Cloud Build.
    buildWorkerPool: Name of the Cloud Build Custom Worker Pool that should be
      used to build the function. The format of this field is
      `projects/{project}/locations/{region}/workerPools/{workerPool}` where
      `{project}` and `{region}` are the project id and region respectively
      where the worker pool is defined and `{workerPool}` is the short name of
      the worker pool. If the project id is not the same as the function, then
      the Cloud Functions Service Agent (`service-@gcf-admin-
      robot.iam.gserviceaccount.com`) must be granted the role Cloud Build
      Custom Workers Builder (`roles/cloudbuild.customworkers.builder`) in the
      project.
    buildpackStack: Specifies one of the Google provided buildpack stacks.
    customStackUri: Deprecated: customization experience was significantly
      revised, making this field obsolete. It was never revealed to public.
      The URL of a customer provided buildpack stack.
    description: User-provided description of a function.
    dockerRegistry: Docker Registry to use for this deployment. If
      unspecified, it defaults to `ARTIFACT_REGISTRY`. If `docker_repository`
      field is specified, this field should either be left unspecified or set
      to `ARTIFACT_REGISTRY`.
    dockerRepository: User managed repository created in Artifact Registry
      optionally with a customer managed encryption key. If specified,
      deployments will use Artifact Registry. If unspecified and the
      deployment is eligible to use Artifact Registry, GCF will create and use
      a repository named 'gcf-artifacts' for every deployed region. This is
      the repository to which the function docker image will be pushed after
      it is built by Cloud Build. It must match the pattern
      `projects/{project}/locations/{location}/repositories/{repository}`.
      Cross-project repositories are not supported. Cross-location
      repositories are not supported. Repository format must be 'DOCKER'.
    entryPoint: The name of the function (as defined in source code) that will
      be executed. Defaults to the resource name suffix (ID of the function),
      if not specified.
    environmentVariables: Environment variables that shall be available during
      function execution.
    eventTrigger: A source that fires events in response to a condition in
      another service.
    httpsTrigger: An HTTPS endpoint type of source that can be triggered via
      URL.
    ingressSettings: The ingress settings for the function, controlling what
      traffic can reach it.
    kmsKeyName: Resource name of a KMS crypto key (managed by the user) used
      to encrypt/decrypt function resources. It must match the pattern `projec
      ts/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto
      _key}`. If specified, you must also provide an artifact registry
      repository using the `docker_repository` field that was created with the
      same KMS crypto key. The following service accounts need to be granted
      the role 'Cloud KMS CryptoKey Encrypter/Decrypter
      (roles/cloudkms.cryptoKeyEncrypterDecrypter)' on the
      Key/KeyRing/Project/Organization (least access preferred). 1. Google
      Cloud Functions service account (service-{project_number}@gcf-admin-
      robot.iam.gserviceaccount.com) - Required to protect the function's
      image. 2. Google Storage service account (service-{project_number}@gs-
      project-accounts.iam.gserviceaccount.com) - Required to protect the
      function's source code. If this service account does not exist,
      deploying a function without a KMS key or retrieving the service agent
      name provisions it. For more information, see
      https://cloud.google.com/storage/docs/projects#service-agents and
      https://cloud.google.com/storage/docs/getting-service-agent#gsutil.
      Google Cloud Functions delegates access to service agents to protect
      function resources in internal projects that are not accessible by the
      end user.
    labels: Labels associated with this Cloud Function.
    maxInstances: The limit on the maximum number of function instances that
      may coexist at a given time. In some cases, such as rapid traffic
      surges, Cloud Functions may, for a short period of time, create more
      instances than the specified max instances limit. If your function
      cannot tolerate this temporary behavior, you may want to factor in a
      safety margin and set a lower max instances value than your function can
      tolerate. See the [Max
      Instances](https://cloud.google.com/functions/docs/max-instances) Guide
      for more details.
    minInstances: A lower bound for the number function instances that may
      coexist at a given time.
    name: A user-defined name of the function. Function names must be unique
      globally and match pattern `projects/*/locations/*/functions/*`
    network: Deprecated: use vpc_connector
    onDeployUpdatePolicy: A OnDeployUpdatePolicy attribute.
    pinnedRuntimeVersionPolicy: A PinnedRuntimeVersionPolicy attribute.
    runDockerfile: Local path to the dockerfile for customizing the base image
      for the worker, located within the source folder.
    runtime: The runtime in which to run the function. Required when deploying
      a new function, optional when updating an existing function. For a
      complete list of possible choices, see the [`gcloud` command reference](
      https://cloud.google.com/sdk/gcloud/reference/functions/deploy#--
      runtime).
    secretEnvironmentVariables: Secret environment variables configuration.
    secretVolumes: Secret volumes configuration.
    serviceAccountEmail: The email of the function's service account. If
      empty, defaults to `{project_id}@appspot.gserviceaccount.com`.
    sourceArchiveUrl: The Google Cloud Storage URL, starting with `gs://`,
      pointing to the zip archive which contains the function.
    sourceRepository: **Beta Feature** The source repository where a function
      is hosted.
    sourceToken: Input only. An identifier for Firebase function sources.
      Disclaimer: This field is only supported for Firebase function
      deployments.
    sourceUploadUrl: The Google Cloud Storage signed URL used for source
      uploading, generated by calling
      [google.cloud.functions.v1.GenerateUploadUrl]. The signature is
      validated on write methods (Create, Update) The signature is stripped
      from the Function object on read methods (Get, List)
    status: Output only. Status of the function deployment.
    timeout: The function execution timeout. Execution is considered failed
      and can be terminated if the function is not completed at the end of the
      timeout period. Defaults to 60 seconds.
    updateTime: Output only. The last update timestamp of a Cloud Function.
    versionId: Output only. The version identifier of the Cloud Function. Each
      deployment attempt results in a new version of a function being created.
    vpcConnector: The VPC Network Connector that this cloud function can
      connect to. It can be either the fully-qualified URI, or the short name
      of the network connector resource. The format of this field is
      `projects/*/locations/*/connectors/*` This field is mutually exclusive
      with `network` field and will eventually replace it. See [the VPC
      documentation](https://cloud.google.com/compute/docs/vpc) for more
      information on connecting Cloud projects.
    vpcConnectorEgressSettings: The egress settings for the connector,
      controlling what traffic is diverted through it.
  """

    class DockerRegistryValueValuesEnum(_messages.Enum):
        """Docker Registry to use for this deployment. If unspecified, it
    defaults to `ARTIFACT_REGISTRY`. If `docker_repository` field is
    specified, this field should either be left unspecified or set to
    `ARTIFACT_REGISTRY`.

    Values:
      DOCKER_REGISTRY_UNSPECIFIED: Unspecified.
      CONTAINER_REGISTRY: Docker images will be stored in multi-regional
        Container Registry repositories named `gcf`.
      ARTIFACT_REGISTRY: Docker images will be stored in regional Artifact
        Registry repositories. By default, GCF will create and use
        repositories named `gcf-artifacts` in every region in which a function
        is deployed. But the repository to use can also be specified by the
        user using the `docker_repository` field.
    """
        DOCKER_REGISTRY_UNSPECIFIED = 0
        CONTAINER_REGISTRY = 1
        ARTIFACT_REGISTRY = 2

    class IngressSettingsValueValuesEnum(_messages.Enum):
        """The ingress settings for the function, controlling what traffic can
    reach it.

    Values:
      INGRESS_SETTINGS_UNSPECIFIED: Unspecified.
      ALLOW_ALL: Allow HTTP traffic from public and private sources.
      ALLOW_INTERNAL_ONLY: Allow HTTP traffic from only private VPC sources.
      ALLOW_INTERNAL_AND_GCLB: Allow HTTP traffic from private VPC sources and
        through GCLB.
    """
        INGRESS_SETTINGS_UNSPECIFIED = 0
        ALLOW_ALL = 1
        ALLOW_INTERNAL_ONLY = 2
        ALLOW_INTERNAL_AND_GCLB = 3

    class StatusValueValuesEnum(_messages.Enum):
        """Output only. Status of the function deployment.

    Values:
      CLOUD_FUNCTION_STATUS_UNSPECIFIED: Not specified. Invalid state.
      ACTIVE: Function has been successfully deployed and is serving.
      OFFLINE: Function deployment failed and the function isn't serving.
      DEPLOY_IN_PROGRESS: Function is being created or updated.
      DELETE_IN_PROGRESS: Function is being deleted.
      UNKNOWN: Function deployment failed and the function serving state is
        undefined. The function should be updated or deleted to move it out of
        this state.
    """
        CLOUD_FUNCTION_STATUS_UNSPECIFIED = 0
        ACTIVE = 1
        OFFLINE = 2
        DEPLOY_IN_PROGRESS = 3
        DELETE_IN_PROGRESS = 4
        UNKNOWN = 5

    class VpcConnectorEgressSettingsValueValuesEnum(_messages.Enum):
        """The egress settings for the connector, controlling what traffic is
    diverted through it.

    Values:
      VPC_CONNECTOR_EGRESS_SETTINGS_UNSPECIFIED: Unspecified.
      PRIVATE_RANGES_ONLY: Use the VPC Access Connector only for private IP
        space from RFC1918.
      ALL_TRAFFIC: Force the use of VPC Access Connector for all egress
        traffic from the function.
    """
        VPC_CONNECTOR_EGRESS_SETTINGS_UNSPECIFIED = 0
        PRIVATE_RANGES_ONLY = 1
        ALL_TRAFFIC = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class BuildEnvironmentVariablesValue(_messages.Message):
        """Build environment variables that shall be available during build time.

    Messages:
      AdditionalProperty: An additional property for a
        BuildEnvironmentVariablesValue object.

    Fields:
      additionalProperties: Additional properties of type
        BuildEnvironmentVariablesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a BuildEnvironmentVariablesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EnvironmentVariablesValue(_messages.Message):
        """Environment variables that shall be available during function
    execution.

    Messages:
      AdditionalProperty: An additional property for a
        EnvironmentVariablesValue object.

    Fields:
      additionalProperties: Additional properties of type
        EnvironmentVariablesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EnvironmentVariablesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels associated with this Cloud Function.

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
    automaticUpdatePolicy = _messages.MessageField('AutomaticUpdatePolicy', 1)
    availableMemoryMb = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    buildDockerfile = _messages.StringField(3)
    buildEnvironmentVariables = _messages.MessageField('BuildEnvironmentVariablesValue', 4)
    buildId = _messages.StringField(5)
    buildName = _messages.StringField(6)
    buildServiceAccount = _messages.StringField(7)
    buildWorkerPool = _messages.StringField(8)
    buildpackStack = _messages.StringField(9)
    customStackUri = _messages.StringField(10)
    description = _messages.StringField(11)
    dockerRegistry = _messages.EnumField('DockerRegistryValueValuesEnum', 12)
    dockerRepository = _messages.StringField(13)
    entryPoint = _messages.StringField(14)
    environmentVariables = _messages.MessageField('EnvironmentVariablesValue', 15)
    eventTrigger = _messages.MessageField('EventTrigger', 16)
    httpsTrigger = _messages.MessageField('HttpsTrigger', 17)
    ingressSettings = _messages.EnumField('IngressSettingsValueValuesEnum', 18)
    kmsKeyName = _messages.StringField(19)
    labels = _messages.MessageField('LabelsValue', 20)
    maxInstances = _messages.IntegerField(21, variant=_messages.Variant.INT32)
    minInstances = _messages.IntegerField(22, variant=_messages.Variant.INT32)
    name = _messages.StringField(23)
    network = _messages.StringField(24)
    onDeployUpdatePolicy = _messages.MessageField('OnDeployUpdatePolicy', 25)
    pinnedRuntimeVersionPolicy = _messages.MessageField('PinnedRuntimeVersionPolicy', 26)
    runDockerfile = _messages.StringField(27)
    runtime = _messages.StringField(28)
    secretEnvironmentVariables = _messages.MessageField('SecretEnvVar', 29, repeated=True)
    secretVolumes = _messages.MessageField('SecretVolume', 30, repeated=True)
    serviceAccountEmail = _messages.StringField(31)
    sourceArchiveUrl = _messages.StringField(32)
    sourceRepository = _messages.MessageField('SourceRepository', 33)
    sourceToken = _messages.StringField(34)
    sourceUploadUrl = _messages.StringField(35)
    status = _messages.EnumField('StatusValueValuesEnum', 36)
    timeout = _messages.StringField(37)
    updateTime = _messages.StringField(38)
    versionId = _messages.IntegerField(39)
    vpcConnector = _messages.StringField(40)
    vpcConnectorEgressSettings = _messages.EnumField('VpcConnectorEgressSettingsValueValuesEnum', 41)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1TrainingInput(_messages.Message):
    """Represents input parameters for a training job. When using the gcloud
  command to submit your training job, you can specify the input parameters as
  command-line arguments and/or in a YAML configuration file referenced from
  the --config command-line argument. For details, see the guide to
  [submitting a training job](/ai-platform/training/docs/training-jobs).

  Enums:
    ScaleTierValueValuesEnum: Required. Specifies the machine types, the
      number of replicas for workers and parameter servers.

  Fields:
    args: Optional. Command-line arguments passed to the training application
      when it starts. If your job uses a custom container, then the arguments
      are passed to the container's `ENTRYPOINT` command.
    enableWebAccess: Optional. Whether you want AI Platform Training to enable
      [interactive shell access](https://cloud.google.com/ai-
      platform/training/docs/monitor-debug-interactive-shell) to training
      containers. If set to `true`, you can access interactive shells at the
      URIs given by TrainingOutput.web_access_uris or
      HyperparameterOutput.web_access_uris (within TrainingOutput.trials).
    encryptionConfig: Optional. Options for using customer-managed encryption
      keys (CMEK) to protect resources created by a training job, instead of
      using Google's default encryption. If this is set, then all resources
      created by the training job will be encrypted with the customer-managed
      encryption key that you specify. [Learn how and when to use CMEK with AI
      Platform Training](/ai-platform/training/docs/cmek).
    evaluatorConfig: Optional. The configuration for evaluators. You should
      only set `evaluatorConfig.acceleratorConfig` if `evaluatorType` is set
      to a Compute Engine machine type. [Learn about restrictions on
      accelerator configurations for training.](/ai-
      platform/training/docs/using-gpus#compute-engine-machine-types-with-gpu)
      Set `evaluatorConfig.imageUri` only if you build a custom image for your
      evaluator. If `evaluatorConfig.imageUri` has not been set, AI Platform
      uses the value of `masterConfig.imageUri`. Learn more about [configuring
      custom containers](/ai-platform/training/docs/distributed-training-
      containers).
    evaluatorCount: Optional. The number of evaluator replicas to use for the
      training job. Each replica in the cluster will be of the type specified
      in `evaluator_type`. This value can only be used when `scale_tier` is
      set to `CUSTOM`. If you set this value, you must also set
      `evaluator_type`. The default value is zero.
    evaluatorType: Optional. Specifies the type of virtual machine to use for
      your training job's evaluator nodes. The supported values are the same
      as those described in the entry for `masterType`. This value must be
      consistent with the category of machine type that `masterType` uses. In
      other words, both must be Compute Engine machine types or both must be
      legacy machine types. This value must be present when `scaleTier` is set
      to `CUSTOM` and `evaluatorCount` is greater than zero.
    hyperparameters: Optional. The set of Hyperparameters to tune.
    jobDir: Optional. A Google Cloud Storage path in which to store training
      outputs and other data needed for training. This path is passed to your
      TensorFlow program as the '--job-dir' command-line argument. The benefit
      of specifying this field is that Cloud ML validates the path for use in
      training.
    masterConfig: Optional. The configuration for your master worker. You
      should only set `masterConfig.acceleratorConfig` if `masterType` is set
      to a Compute Engine machine type. Learn about [restrictions on
      accelerator configurations for training.](/ai-
      platform/training/docs/using-gpus#compute-engine-machine-types-with-gpu)
      Set `masterConfig.imageUri` only if you build a custom image. Only one
      of `masterConfig.imageUri` and `runtimeVersion` should be set. Learn
      more about [configuring custom containers](/ai-
      platform/training/docs/distributed-training-containers).
    masterType: Optional. Specifies the type of virtual machine to use for
      your training job's master worker. You must specify this field when
      `scaleTier` is set to `CUSTOM`. You can use certain Compute Engine
      machine types directly in this field. See the [list of compatible
      Compute Engine machine types](/ai-platform/training/docs/machine-
      types#compute-engine-machine-types). Alternatively, you can use the
      certain legacy machine types in this field. See the [list of legacy
      machine types](/ai-platform/training/docs/machine-types#legacy-machine-
      types). Finally, if you want to use a TPU for training, specify
      `cloud_tpu` in this field. Learn more about the [special configuration
      options for training with TPUs](/ai-platform/training/docs/using-
      tpus#configuring_a_custom_tpu_machine).
    nasJobSpec: Optional. The spec of a Neural Architecture Search (NAS) job.
    network: Optional. The full name of the [Compute Engine
      network](/vpc/docs/vpc) to which the Job is peered. For example,
      `projects/12345/global/networks/myVPC`. The format of this field is
      `projects/{project}/global/networks/{network}`, where {project} is a
      project number (like `12345`) and {network} is network name. Private
      services access must already be configured for the network. If left
      unspecified, the Job is not peered with any network. [Learn about using
      VPC Network Peering.](/ai-platform/training/docs/vpc-peering).
    packageUris: Required. The Google Cloud Storage location of the packages
      with the training program and any additional dependencies. The maximum
      number of package URIs is 100.
    parameterServerConfig: Optional. The configuration for parameter servers.
      You should only set `parameterServerConfig.acceleratorConfig` if
      `parameterServerType` is set to a Compute Engine machine type. [Learn
      about restrictions on accelerator configurations for training.](/ai-
      platform/training/docs/using-gpus#compute-engine-machine-types-with-gpu)
      Set `parameterServerConfig.imageUri` only if you build a custom image
      for your parameter server. If `parameterServerConfig.imageUri` has not
      been set, AI Platform uses the value of `masterConfig.imageUri`. Learn
      more about [configuring custom containers](/ai-
      platform/training/docs/distributed-training-containers).
    parameterServerCount: Optional. The number of parameter server replicas to
      use for the training job. Each replica in the cluster will be of the
      type specified in `parameter_server_type`. This value can only be used
      when `scale_tier` is set to `CUSTOM`. If you set this value, you must
      also set `parameter_server_type`. The default value is zero.
    parameterServerType: Optional. Specifies the type of virtual machine to
      use for your training job's parameter server. The supported values are
      the same as those described in the entry for `master_type`. This value
      must be consistent with the category of machine type that `masterType`
      uses. In other words, both must be Compute Engine machine types or both
      must be legacy machine types. This value must be present when
      `scaleTier` is set to `CUSTOM` and `parameter_server_count` is greater
      than zero.
    pythonModule: Required. The Python module name to run after installing the
      packages.
    pythonVersion: Optional. The version of Python used in training. You must
      either specify this field or specify `masterConfig.imageUri`. The
      following Python versions are available: * Python '3.7' is available
      when `runtime_version` is set to '1.15' or later. * Python '3.5' is
      available when `runtime_version` is set to a version from '1.4' to
      '1.14'. * Python '2.7' is available when `runtime_version` is set to
      '1.15' or earlier. Read more about the Python versions available for
      [each runtime version](/ml-engine/docs/runtime-version-list).
    region: Required. The region to run the training job in. See the
      [available regions](/ai-platform/training/docs/regions) for AI Platform
      Training.
    runtimeVersion: Optional. The AI Platform runtime version to use for
      training. You must either specify this field or specify
      `masterConfig.imageUri`. For more information, see the [runtime version
      list](/ai-platform/training/docs/runtime-version-list) and learn [how to
      manage runtime versions](/ai-platform/training/docs/versioning).
    scaleTier: Required. Specifies the machine types, the number of replicas
      for workers and parameter servers.
    scheduling: Optional. Scheduling options for a training job.
    serviceAccount: Optional. The email address of a service account to use
      when running the training appplication. You must have the
      `iam.serviceAccounts.actAs` permission for the specified service
      account. In addition, the AI Platform Training Google-managed service
      account must have the `roles/iam.serviceAccountAdmin` role for the
      specified service account. [Learn more about configuring a service
      account.](/ai-platform/training/docs/custom-service-account) If not
      specified, the AI Platform Training Google-managed service account is
      used by default.
    useChiefInTfConfig: Optional. Use `chief` instead of `master` in the
      `TF_CONFIG` environment variable when training with a custom container.
      Defaults to `false`. [Learn more about this field.](/ai-
      platform/training/docs/distributed-training-details#chief-versus-master)
      This field has no effect for training jobs that don't use a custom
      container.
    workerConfig: Optional. The configuration for workers. You should only set
      `workerConfig.acceleratorConfig` if `workerType` is set to a Compute
      Engine machine type. [Learn about restrictions on accelerator
      configurations for training.](/ai-platform/training/docs/using-
      gpus#compute-engine-machine-types-with-gpu) Set `workerConfig.imageUri`
      only if you build a custom image for your worker. If
      `workerConfig.imageUri` has not been set, AI Platform uses the value of
      `masterConfig.imageUri`. Learn more about [configuring custom
      containers](/ai-platform/training/docs/distributed-training-containers).
    workerCount: Optional. The number of worker replicas to use for the
      training job. Each replica in the cluster will be of the type specified
      in `worker_type`. This value can only be used when `scale_tier` is set
      to `CUSTOM`. If you set this value, you must also set `worker_type`. The
      default value is zero.
    workerType: Optional. Specifies the type of virtual machine to use for
      your training job's worker nodes. The supported values are the same as
      those described in the entry for `masterType`. This value must be
      consistent with the category of machine type that `masterType` uses. In
      other words, both must be Compute Engine machine types or both must be
      legacy machine types. If you use `cloud_tpu` for this value, see special
      instructions for [configuring a custom TPU machine](/ml-
      engine/docs/tensorflow/using-tpus#configuring_a_custom_tpu_machine).
      This value must be present when `scaleTier` is set to `CUSTOM` and
      `workerCount` is greater than zero.
  """

    class ScaleTierValueValuesEnum(_messages.Enum):
        """Required. Specifies the machine types, the number of replicas for
    workers and parameter servers.

    Values:
      BASIC: A single worker instance. This tier is suitable for learning how
        to use Cloud ML, and for experimenting with new models using small
        datasets.
      STANDARD_1: Many workers and a few parameter servers.
      PREMIUM_1: A large number of workers with many parameter servers.
      BASIC_GPU: A single worker instance [with a GPU](/ai-
        platform/training/docs/using-gpus).
      BASIC_TPU: A single worker instance with a [Cloud TPU](/ml-
        engine/docs/tensorflow/using-tpus).
      CUSTOM: The CUSTOM tier is not a set tier, but rather enables you to use
        your own cluster specification. When you use this tier, set values to
        configure your processing cluster according to these guidelines: * You
        _must_ set `TrainingInput.masterType` to specify the type of machine
        to use for your master node. This is the only required setting. * You
        _may_ set `TrainingInput.workerCount` to specify the number of workers
        to use. If you specify one or more workers, you _must_ also set
        `TrainingInput.workerType` to specify the type of machine to use for
        your worker nodes. * You _may_ set
        `TrainingInput.parameterServerCount` to specify the number of
        parameter servers to use. If you specify one or more parameter
        servers, you _must_ also set `TrainingInput.parameterServerType` to
        specify the type of machine to use for your parameter servers. Note
        that all of your workers must use the same machine type, which can be
        different from your parameter server type and master type. Your
        parameter servers must likewise use the same machine type, which can
        be different from your worker type and master type.
    """
        BASIC = 0
        STANDARD_1 = 1
        PREMIUM_1 = 2
        BASIC_GPU = 3
        BASIC_TPU = 4
        CUSTOM = 5
    args = _messages.StringField(1, repeated=True)
    enableWebAccess = _messages.BooleanField(2)
    encryptionConfig = _messages.MessageField('GoogleCloudMlV1EncryptionConfig', 3)
    evaluatorConfig = _messages.MessageField('GoogleCloudMlV1ReplicaConfig', 4)
    evaluatorCount = _messages.IntegerField(5)
    evaluatorType = _messages.StringField(6)
    hyperparameters = _messages.MessageField('GoogleCloudMlV1HyperparameterSpec', 7)
    jobDir = _messages.StringField(8)
    masterConfig = _messages.MessageField('GoogleCloudMlV1ReplicaConfig', 9)
    masterType = _messages.StringField(10)
    nasJobSpec = _messages.MessageField('GoogleCloudMlV1NasSpec', 11)
    network = _messages.StringField(12)
    packageUris = _messages.StringField(13, repeated=True)
    parameterServerConfig = _messages.MessageField('GoogleCloudMlV1ReplicaConfig', 14)
    parameterServerCount = _messages.IntegerField(15)
    parameterServerType = _messages.StringField(16)
    pythonModule = _messages.StringField(17)
    pythonVersion = _messages.StringField(18)
    region = _messages.StringField(19)
    runtimeVersion = _messages.StringField(20)
    scaleTier = _messages.EnumField('ScaleTierValueValuesEnum', 21)
    scheduling = _messages.MessageField('GoogleCloudMlV1Scheduling', 22)
    serviceAccount = _messages.StringField(23)
    useChiefInTfConfig = _messages.BooleanField(24)
    workerConfig = _messages.MessageField('GoogleCloudMlV1ReplicaConfig', 25)
    workerCount = _messages.IntegerField(26)
    workerType = _messages.StringField(27)
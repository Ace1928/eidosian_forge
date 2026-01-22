from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1CustomJobSpec(_messages.Message):
    """Represents the spec of a CustomJob.

  Fields:
    baseOutputDirectory: The Cloud Storage location to store the output of
      this CustomJob or HyperparameterTuningJob. For HyperparameterTuningJob,
      the baseOutputDirectory of each child CustomJob backing a Trial is set
      to a subdirectory of name id under its parent HyperparameterTuningJob's
      baseOutputDirectory. The following Vertex AI environment variables will
      be passed to containers or python modules when this field is set: For
      CustomJob: * AIP_MODEL_DIR = `/model/` * AIP_CHECKPOINT_DIR =
      `/checkpoints/` * AIP_TENSORBOARD_LOG_DIR = `/logs/` For CustomJob
      backing a Trial of HyperparameterTuningJob: * AIP_MODEL_DIR = `//model/`
      * AIP_CHECKPOINT_DIR = `//checkpoints/` * AIP_TENSORBOARD_LOG_DIR =
      `//logs/`
    enableDashboardAccess: Optional. Whether you want Vertex AI to enable
      access to the customized dashboard in training chief container. If set
      to `true`, you can access the dashboard at the URIs given by
      CustomJob.web_access_uris or Trial.web_access_uris (within
      HyperparameterTuningJob.trials).
    enableWebAccess: Optional. Whether you want Vertex AI to enable
      [interactive shell access](https://cloud.google.com/vertex-
      ai/docs/training/monitor-debug-interactive-shell) to training
      containers. If set to `true`, you can access interactive shells at the
      URIs given by CustomJob.web_access_uris or Trial.web_access_uris (within
      HyperparameterTuningJob.trials).
    experiment: Optional. The Experiment associated with this job. Format: `pr
      ojects/{project}/locations/{location}/metadataStores/{metadataStores}/co
      ntexts/{experiment-name}`
    experimentRun: Optional. The Experiment Run associated with this job.
      Format: `projects/{project}/locations/{location}/metadataStores/{metadat
      aStores}/contexts/{experiment-name}-{experiment-run-name}`
    models: Optional. The name of the Model resources for which to generate a
      mapping to artifact URIs. Applicable only to some of the Google-provided
      custom jobs. Format:
      `projects/{project}/locations/{location}/models/{model}` In order to
      retrieve a specific version of the model, also provide the version ID or
      version alias. Example:
      `projects/{project}/locations/{location}/models/{model}@2` or
      `projects/{project}/locations/{location}/models/{model}@golden` If no
      version ID or alias is specified, the "default" version will be
      returned. The "default" version alias is created for the first version
      of the model, and can be moved to other versions later on. There will be
      exactly one default version.
    network: Optional. The full name of the Compute Engine
      [network](/compute/docs/networks-and-firewalls#networks) to which the
      Job should be peered. For example,
      `projects/12345/global/networks/myVPC`.
      [Format](/compute/docs/reference/rest/v1/networks/insert) is of the form
      `projects/{project}/global/networks/{network}`. Where {project} is a
      project number, as in `12345`, and {network} is a network name. To
      specify this field, you must have already [configured VPC Network
      Peering for Vertex AI](https://cloud.google.com/vertex-
      ai/docs/general/vpc-peering). If this field is left unspecified, the job
      is not peered with any network.
    persistentResourceId: Optional. The ID of the PersistentResource in the
      same Project and Location which to run If this is specified, the job
      will be run on existing machines held by the PersistentResource instead
      of on-demand short-live machines. The network and CMEK configs on the
      job should be consistent with those on the PersistentResource,
      otherwise, the job will be rejected.
    protectedArtifactLocationId: The ID of the location to store protected
      artifacts. e.g. us-central1. Populate only when the location is
      different than CustomJob location. List of supported locations:
      https://cloud.google.com/vertex-ai/docs/general/locations
    reservedIpRanges: Optional. A list of names for the reserved ip ranges
      under the VPC network that can be used for this job. If set, we will
      deploy the job within the provided ip ranges. Otherwise, the job will be
      deployed to any ip ranges under the provided VPC network. Example:
      ['vertex-ai-ip-range'].
    scheduling: Scheduling options for a CustomJob.
    serviceAccount: Specifies the service account for workload run-as account.
      Users submitting jobs must have act-as permission on this run-as
      account. If unspecified, the [Vertex AI Custom Code Service
      Agent](https://cloud.google.com/vertex-ai/docs/general/access-
      control#service-agents) for the CustomJob's project is used.
    tensorboard: Optional. The name of a Vertex AI Tensorboard resource to
      which this CustomJob will upload Tensorboard logs. Format:
      `projects/{project}/locations/{location}/tensorboards/{tensorboard}`
    workerPoolSpecs: Required. The spec of the worker pools including machine
      type and Docker image. All worker pools except the first one are
      optional and can be skipped by providing an empty value.
  """
    baseOutputDirectory = _messages.MessageField('GoogleCloudAiplatformV1GcsDestination', 1)
    enableDashboardAccess = _messages.BooleanField(2)
    enableWebAccess = _messages.BooleanField(3)
    experiment = _messages.StringField(4)
    experimentRun = _messages.StringField(5)
    models = _messages.StringField(6, repeated=True)
    network = _messages.StringField(7)
    persistentResourceId = _messages.StringField(8)
    protectedArtifactLocationId = _messages.StringField(9)
    reservedIpRanges = _messages.StringField(10, repeated=True)
    scheduling = _messages.MessageField('GoogleCloudAiplatformV1Scheduling', 11)
    serviceAccount = _messages.StringField(12)
    tensorboard = _messages.StringField(13)
    workerPoolSpecs = _messages.MessageField('GoogleCloudAiplatformV1WorkerPoolSpec', 14, repeated=True)
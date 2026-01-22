from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ReplicaConfig(_messages.Message):
    """Represents the configuration for a replica in a cluster.

  Fields:
    acceleratorConfig: Represents the type and number of accelerators used by
      the replica. [Learn about restrictions on accelerator configurations for
      training.](/ai-platform/training/docs/using-gpus#compute-engine-machine-
      types-with-gpu)
    containerArgs: Arguments to the entrypoint command. The following rules
      apply for container_command and container_args: - If you do not supply
      command or args: The defaults defined in the Docker image are used. - If
      you supply a command but no args: The default EntryPoint and the default
      Cmd defined in the Docker image are ignored. Your command is run without
      any arguments. - If you supply only args: The default Entrypoint defined
      in the Docker image is run with the args that you supplied. - If you
      supply a command and args: The default Entrypoint and the default Cmd
      defined in the Docker image are ignored. Your command is run with your
      args. It cannot be set if custom container image is not provided. Note
      that this field and [TrainingInput.args] are mutually exclusive, i.e.,
      both cannot be set at the same time.
    containerCommand: The command with which the replica's custom container is
      run. If provided, it will override default ENTRYPOINT of the docker
      image. If not provided, the docker image's ENTRYPOINT is used. It cannot
      be set if custom container image is not provided. Note that this field
      and [TrainingInput.args] are mutually exclusive, i.e., both cannot be
      set at the same time.
    diskConfig: Represents the configuration of disk options.
    imageUri: The Docker image to run on the replica. This image must be in
      Container Registry. Learn more about [configuring custom
      containers](/ai-platform/training/docs/distributed-training-containers).
    tpuTfVersion: The AI Platform runtime version that includes a TensorFlow
      version matching the one used in the custom container. This field is
      required if the replica is a TPU worker that uses a custom container.
      Otherwise, do not specify this field. This must be a [runtime version
      that currently supports training with TPUs](/ml-
      engine/docs/tensorflow/runtime-version-list#tpu-support). Note that the
      version of TensorFlow included in a runtime version may differ from the
      numbering of the runtime version itself, because it may have a different
      [patch version](https://www.tensorflow.org/guide/version_compat#semantic
      _versioning_20). In this field, you must specify the runtime version
      (TensorFlow minor version). For example, if your custom container runs
      TensorFlow `1.x.y`, specify `1.x`.
  """
    acceleratorConfig = _messages.MessageField('GoogleCloudMlV1AcceleratorConfig', 1)
    containerArgs = _messages.StringField(2, repeated=True)
    containerCommand = _messages.StringField(3, repeated=True)
    diskConfig = _messages.MessageField('GoogleCloudMlV1DiskConfig', 4)
    imageUri = _messages.StringField(5)
    tpuTfVersion = _messages.StringField(6)
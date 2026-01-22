from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2Container(_messages.Message):
    """A single application container. This specifies both the container to
  run, the command to run in the container and the arguments to supply to it.
  Note that additional arguments can be supplied by the system to the
  container at runtime.

  Fields:
    args: Arguments to the entrypoint. The docker image's CMD is used if this
      is not provided.
    command: Entrypoint array. Not executed within a shell. The docker image's
      ENTRYPOINT is used if this is not provided.
    dependsOn: Names of the containers that must start before this container.
    env: List of environment variables to set in the container.
    image: Required. Name of the container image in Dockerhub, Google Artifact
      Registry, or Google Container Registry. If the host is not provided,
      Dockerhub is assumed.
    livenessProbe: Periodic probe of container liveness. Container will be
      restarted if the probe fails.
    name: Name of the container specified as a DNS_LABEL (RFC 1123).
    ports: List of ports to expose from the container. Only a single port can
      be specified. The specified ports must be listening on all interfaces
      (0.0.0.0) within the container to be accessible. If omitted, a port
      number will be chosen and passed to the container through the PORT
      environment variable for the container to listen on.
    resources: Compute Resource requirements by this container.
    startupProbe: Startup probe of application within the container. All other
      probes are disabled if a startup probe is provided, until it succeeds.
      Container will not be added to service endpoints if the probe fails.
    volumeMounts: Volume to mount into the container's filesystem.
    workingDir: Container's working directory. If not specified, the container
      runtime's default will be used, which might be configured in the
      container image.
  """
    args = _messages.StringField(1, repeated=True)
    command = _messages.StringField(2, repeated=True)
    dependsOn = _messages.StringField(3, repeated=True)
    env = _messages.MessageField('GoogleCloudRunV2EnvVar', 4, repeated=True)
    image = _messages.StringField(5)
    livenessProbe = _messages.MessageField('GoogleCloudRunV2Probe', 6)
    name = _messages.StringField(7)
    ports = _messages.MessageField('GoogleCloudRunV2ContainerPort', 8, repeated=True)
    resources = _messages.MessageField('GoogleCloudRunV2ResourceRequirements', 9)
    startupProbe = _messages.MessageField('GoogleCloudRunV2Probe', 10)
    volumeMounts = _messages.MessageField('GoogleCloudRunV2VolumeMount', 11, repeated=True)
    workingDir = _messages.StringField(12)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Sidecar(_messages.Message):
    """Sidecars run alongside the Task's step containers.

  Fields:
    args: Arguments to the entrypoint.
    command: Entrypoint array.
    env: List of environment variables to set in the container.
    image: Docker image name.
    name: Name of the Sidecar.
    readinessProbe: Optional. Periodic probe of Sidecar service readiness.
      Container will be removed from service endpoints if the probe fails.
      Cannot be updated. More info:
      https://kubernetes.io/docs/concepts/workloads/pods/pod-
      lifecycle#container-probes +optional
    script: The contents of an executable file to execute.
    securityContext: Optional. Security options the container should be run
      with.
    volumeMounts: Pod volumes to mount into the container's filesystem.
    workingDir: Container's working directory.
  """
    args = _messages.StringField(1, repeated=True)
    command = _messages.StringField(2, repeated=True)
    env = _messages.MessageField('EnvVar', 3, repeated=True)
    image = _messages.StringField(4)
    name = _messages.StringField(5)
    readinessProbe = _messages.MessageField('Probe', 6)
    script = _messages.StringField(7)
    securityContext = _messages.MessageField('SecurityContext', 8)
    volumeMounts = _messages.MessageField('VolumeMount', 9, repeated=True)
    workingDir = _messages.StringField(10)
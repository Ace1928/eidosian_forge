from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ContainerSpec(_messages.Message):
    """The spec of a Container.

  Fields:
    args: The arguments to be passed when starting the container.
    command: The command to be invoked when the container is started. It
      overrides the entrypoint instruction in Dockerfile when provided.
    env: Environment variables to be passed to the container. Maximum limit is
      100.
    imageUri: Required. The URI of a container image in the Container Registry
      that is to be run on each worker replica.
  """
    args = _messages.StringField(1, repeated=True)
    command = _messages.StringField(2, repeated=True)
    env = _messages.MessageField('GoogleCloudAiplatformV1EnvVar', 3, repeated=True)
    imageUri = _messages.StringField(4)
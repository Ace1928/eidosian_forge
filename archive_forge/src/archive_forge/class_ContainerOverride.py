from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerOverride(_messages.Message):
    """Per container override specification.

  Fields:
    args: Arguments to the entrypoint. The specified arguments replace and
      override any existing entrypoint arguments. Must be empty if
      `clear_args` is set to true.
    clearArgs: Optional. Set to True to clear all existing arguments.
    env: List of environment variables to set in the container. All specified
      environment variables are merged with existing environment variables.
      When the specified environment variables exist, these values override
      any existing values.
    name: The name of the container specified as a DNS_LABEL.
  """
    args = _messages.StringField(1, repeated=True)
    clearArgs = _messages.BooleanField(2)
    env = _messages.MessageField('EnvVar', 3, repeated=True)
    name = _messages.StringField(4)
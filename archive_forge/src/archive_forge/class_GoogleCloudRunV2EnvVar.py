from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2EnvVar(_messages.Message):
    """EnvVar represents an environment variable present in a Container.

  Fields:
    name: Required. Name of the environment variable. Must not exceed 32768
      characters.
    value: Variable references $(VAR_NAME) are expanded using the previous
      defined environment variables in the container and any route environment
      variables. If a variable cannot be resolved, the reference in the input
      string will be unchanged. The $(VAR_NAME) syntax can be escaped with a
      double $$, ie: $$(VAR_NAME). Escaped references will never be expanded,
      regardless of whether the variable exists or not. Defaults to "", and
      the maximum length is 32768 bytes.
    valueSource: Source for the environment variable's value.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)
    valueSource = _messages.MessageField('GoogleCloudRunV2EnvVarSource', 3)
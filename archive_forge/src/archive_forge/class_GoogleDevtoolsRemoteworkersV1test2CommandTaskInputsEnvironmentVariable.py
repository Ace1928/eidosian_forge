from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemoteworkersV1test2CommandTaskInputsEnvironmentVariable(_messages.Message):
    """An environment variable required by this task.

  Fields:
    name: The envvar name.
    value: The envvar value.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)
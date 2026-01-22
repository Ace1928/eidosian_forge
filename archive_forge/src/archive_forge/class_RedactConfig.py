from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedactConfig(_messages.Message):
    """Defines how to redact sensitive values. Default behavior is erase. For
  example, "My name is Jane." becomes "My name is ."
  """
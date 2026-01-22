from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class VulnerableParameters(_messages.Message):
    """Information about vulnerable request parameters.

  Fields:
    parameterNames: The vulnerable parameter names.
  """
    parameterNames = _messages.StringField(1, repeated=True)
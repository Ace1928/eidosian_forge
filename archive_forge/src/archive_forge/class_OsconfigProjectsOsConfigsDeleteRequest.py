from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsOsConfigsDeleteRequest(_messages.Message):
    """A OsconfigProjectsOsConfigsDeleteRequest object.

  Fields:
    name: The resource name of the OsConfig.
  """
    name = _messages.StringField(1, required=True)
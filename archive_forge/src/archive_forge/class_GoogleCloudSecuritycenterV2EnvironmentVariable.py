from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2EnvironmentVariable(_messages.Message):
    """A name-value pair representing an environment variable used in an
  operating system process.

  Fields:
    name: Environment variable name as a JSON encoded string.
    val: Environment variable value as a JSON encoded string.
  """
    name = _messages.StringField(1)
    val = _messages.StringField(2)
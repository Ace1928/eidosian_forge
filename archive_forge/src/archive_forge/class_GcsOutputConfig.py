from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcsOutputConfig(_messages.Message):
    """Output configurations for Cloud Storage.

  Fields:
    uri: The Cloud Storage URI prefix with which recognition results will be
      written.
  """
    uri = _messages.StringField(1)
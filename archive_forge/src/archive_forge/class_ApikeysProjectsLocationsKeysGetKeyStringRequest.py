from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApikeysProjectsLocationsKeysGetKeyStringRequest(_messages.Message):
    """A ApikeysProjectsLocationsKeysGetKeyStringRequest object.

  Fields:
    name: Required. The resource name of the API key to be retrieved.
  """
    name = _messages.StringField(1, required=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApikeysProjectsKeysCreateRequest(_messages.Message):
    """A ApikeysProjectsKeysCreateRequest object.

  Fields:
    parent: Required. The project for which this API key will be created.
    v2alpha1ApiKey: A V2alpha1ApiKey resource to be passed as the request
      body.
  """
    parent = _messages.StringField(1, required=True)
    v2alpha1ApiKey = _messages.MessageField('V2alpha1ApiKey', 2)
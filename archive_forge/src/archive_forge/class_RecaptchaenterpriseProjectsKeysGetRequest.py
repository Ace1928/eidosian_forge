from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsKeysGetRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsKeysGetRequest object.

  Fields:
    name: Required. The name of the requested key, in the format
      `projects/{project}/keys/{key}`.
  """
    name = _messages.StringField(1, required=True)
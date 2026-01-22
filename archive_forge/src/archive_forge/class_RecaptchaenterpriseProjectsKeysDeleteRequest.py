from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsKeysDeleteRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsKeysDeleteRequest object.

  Fields:
    name: Required. The name of the key to be deleted, in the format
      `projects/{project}/keys/{key}`.
  """
    name = _messages.StringField(1, required=True)
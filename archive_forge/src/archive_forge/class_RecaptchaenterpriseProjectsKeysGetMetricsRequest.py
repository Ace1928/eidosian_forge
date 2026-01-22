from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsKeysGetMetricsRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsKeysGetMetricsRequest object.

  Fields:
    name: Required. The name of the requested metrics, in the format
      `projects/{project}/keys/{key}/metrics`.
  """
    name = _messages.StringField(1, required=True)
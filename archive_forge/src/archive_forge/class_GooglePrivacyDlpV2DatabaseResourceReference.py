from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DatabaseResourceReference(_messages.Message):
    """Identifies a single database resource, like a table within a database.

  Fields:
    instance: Required. The instance where this resource is located. For
      example: Cloud SQL's instance id.
    projectId: Required. If within a project-level config, then this must
      match the config's project id.
  """
    instance = _messages.StringField(1)
    projectId = _messages.StringField(2)
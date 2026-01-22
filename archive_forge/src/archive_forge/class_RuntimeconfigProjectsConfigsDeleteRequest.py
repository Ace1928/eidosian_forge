from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeconfigProjectsConfigsDeleteRequest(_messages.Message):
    """A RuntimeconfigProjectsConfigsDeleteRequest object.

  Fields:
    name: The RuntimeConfig resource to delete, in the format:
      `projects/[PROJECT_ID]/configs/[CONFIG_NAME]`
  """
    name = _messages.StringField(1, required=True)
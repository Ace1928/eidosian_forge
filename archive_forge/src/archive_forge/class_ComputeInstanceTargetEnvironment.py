from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstanceTargetEnvironment(_messages.Message):
    """ComputeInstanceTargetEnvironment represents Compute Engine target
  environment to be used during restore.

  Fields:
    project: Required. Name of the restore target project in the format
      `projects/{project_id}.
    zone: Required. The zone of the Compute Engine instance.
  """
    project = _messages.StringField(1)
    zone = _messages.StringField(2)
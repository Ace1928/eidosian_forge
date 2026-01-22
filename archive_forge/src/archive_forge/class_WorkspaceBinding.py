from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkspaceBinding(_messages.Message):
    """WorkspaceBinding maps a workspace to a Volume. PipelineRef can be used
  to refer to a specific instance of a Pipeline.

  Fields:
    name: Name of the workspace.
    secret: Secret Volume Source.
    subPath: Optional. SubPath is optionally a directory on the volume which
      should be used for this binding (i.e. the volume will be mounted at this
      sub directory). +optional
    volumeClaim: Volume claim that will be created in the same namespace.
      Deprecated, do not use for workloads that don't use workerpools.
  """
    name = _messages.StringField(1)
    secret = _messages.MessageField('SecretVolumeSource', 2)
    subPath = _messages.StringField(3)
    volumeClaim = _messages.MessageField('VolumeClaim', 4)
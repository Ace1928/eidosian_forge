from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ErrorResource(_messages.Message):
    """Model for a config file in the git repo with an associated Sync error

  Fields:
    resourceGvk: Group/version/kind of the resource that is causing an error
    resourceName: Metadata name of the resource that is causing an error
    resourceNamespace: Namespace of the resource that is causing an error
    sourcePath: Path in the git repo of the erroneous config
  """
    resourceGvk = _messages.MessageField('GroupVersionKind', 1)
    resourceName = _messages.StringField(2)
    resourceNamespace = _messages.StringField(3)
    sourcePath = _messages.StringField(4)
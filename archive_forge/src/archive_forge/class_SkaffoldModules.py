from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SkaffoldModules(_messages.Message):
    """Skaffold Config modules and their remote source.

  Fields:
    configs: Optional. The Skaffold Config modules to use from the specified
      source.
    git: Remote git repository containing the Skaffold Config modules.
    googleCloudStorage: Cloud Storage bucket containing the Skaffold Config
      modules.
  """
    configs = _messages.StringField(1, repeated=True)
    git = _messages.MessageField('SkaffoldGitSource', 2)
    googleCloudStorage = _messages.MessageField('SkaffoldGCSSource', 3)
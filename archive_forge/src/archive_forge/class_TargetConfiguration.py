from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetConfiguration(_messages.Message):
    """A TargetConfiguration object.

  Fields:
    config: The configuration to use for this deployment.
    imports: Specifies any files to import for this configuration. This can be
      used to import templates or other files. For example, you might import a
      text file in order to use the file in a template.
  """
    config = _messages.MessageField('ConfigFile', 1)
    imports = _messages.MessageField('ImportFile', 2, repeated=True)
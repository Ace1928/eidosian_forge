from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SshScript(_messages.Message):
    """Response message for 'GenerateSshScript' request.

  Fields:
    script: The ssh configuration script.
  """
    script = _messages.StringField(1)
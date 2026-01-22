from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Entrypoint(_messages.Message):
    """The entrypoint for the application.

  Fields:
    shell: The format should be a shell command that can be fed to bash -c.
  """
    shell = _messages.StringField(1)
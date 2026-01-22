from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportStatefileRequest(_messages.Message):
    """A request to import a state file passed to a 'ImportStatefile' call.

  Fields:
    lockId: Required. Lock ID of the lock file to verify that the user who is
      importing the state file previously locked the Deployment.
  """
    lockId = _messages.IntegerField(1)
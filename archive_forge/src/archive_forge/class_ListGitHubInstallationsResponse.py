from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListGitHubInstallationsResponse(_messages.Message):
    """RPC response object accepted by the ListGitHubInstallations RPC method.

  Fields:
    installations: Installations matching the requested installation ID.
  """
    installations = _messages.MessageField('Installation', 1, repeated=True)
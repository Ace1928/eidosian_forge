from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsNfsSharesRenameRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsNfsSharesRenameRequest object.

  Fields:
    name: Required. The `name` field is used to identify the nfsshare. Format:
      projects/{project}/locations/{location}/nfsshares/{nfsshare}
    renameNfsShareRequest: A RenameNfsShareRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    renameNfsShareRequest = _messages.MessageField('RenameNfsShareRequest', 2)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsNetworksRenameRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsNetworksRenameRequest object.

  Fields:
    name: Required. The `name` field is used to identify the network. Format:
      projects/{project}/locations/{location}/networks/{network}
    renameNetworkRequest: A RenameNetworkRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    renameNetworkRequest = _messages.MessageField('RenameNetworkRequest', 2)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListBlockchainValidatorConfigsResponse(_messages.Message):
    """A message representing all blockchain validator configs in the project.

  Fields:
    blockchainValidatorConfigs: The validator configurations defined within
      the project and location.
    nextPageToken: A token identifying a page of results the server should
      return.
    unreachable: Locations that could not be reached.
  """
    blockchainValidatorConfigs = _messages.MessageField('BlockchainValidatorConfig', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)
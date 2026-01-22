from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsMigrationJobsDemoteDestinationRequest(_messages.Message):
    """A DatamigrationProjectsLocationsMigrationJobsDemoteDestinationRequest
  object.

  Fields:
    demoteDestinationRequest: A DemoteDestinationRequest resource to be passed
      as the request body.
    name: Name of the migration job resource to demote its destination.
  """
    demoteDestinationRequest = _messages.MessageField('DemoteDestinationRequest', 1)
    name = _messages.StringField(2, required=True)
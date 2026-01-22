from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageinsightsProjectsLocationsDatasetConfigsUnlinkDatasetRequest(_messages.Message):
    """A StorageinsightsProjectsLocationsDatasetConfigsUnlinkDatasetRequest
  object.

  Fields:
    name: Required. Name of the resource
    unlinkDatasetRequest: A UnlinkDatasetRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    unlinkDatasetRequest = _messages.MessageField('UnlinkDatasetRequest', 2)
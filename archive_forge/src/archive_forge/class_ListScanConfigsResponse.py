from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListScanConfigsResponse(_messages.Message):
    """A list of scan configs for the project.

  Fields:
    nextPageToken: A page token to pass in order to get more scan configs.
    scanConfigs: The set of scan configs.
  """
    nextPageToken = _messages.StringField(1)
    scanConfigs = _messages.MessageField('ScanConfig', 2, repeated=True)
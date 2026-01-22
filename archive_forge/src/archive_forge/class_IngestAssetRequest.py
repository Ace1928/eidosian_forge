from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IngestAssetRequest(_messages.Message):
    """Request to ingest an other-cloud asset.

  Fields:
    asset: The other-cloud asset to be ingested.
  """
    asset = _messages.MessageField('OtherCloudAssetEvent', 1)
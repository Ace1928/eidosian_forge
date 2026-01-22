from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetIngestAssetRequest(_messages.Message):
    """A CloudassetIngestAssetRequest object.

  Fields:
    closestCrmAncestor: The closest Google Cloud Resource Manager ancestor of
      this asset. The format will be: organizations/, or folders/, or
      projects/
    ingestAssetRequest: A IngestAssetRequest resource to be passed as the
      request body.
  """
    closestCrmAncestor = _messages.StringField(1, required=True)
    ingestAssetRequest = _messages.MessageField('IngestAssetRequest', 2)
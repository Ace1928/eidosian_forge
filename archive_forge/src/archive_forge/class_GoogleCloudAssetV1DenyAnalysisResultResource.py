from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1DenyAnalysisResultResource(_messages.Message):
    """A Google Cloud resource under analysis.

  Fields:
    fullResourceName: The [full resource name](https://cloud.google.com/asset-
      inventory/docs/resource-name-format)
  """
    fullResourceName = _messages.StringField(1)
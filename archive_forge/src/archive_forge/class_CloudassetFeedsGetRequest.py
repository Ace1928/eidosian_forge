from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetFeedsGetRequest(_messages.Message):
    """A CloudassetFeedsGetRequest object.

  Fields:
    name: Required. The name of the Feed and it must be in the format of:
      projects/project_number/feeds/feed_id
      folders/folder_number/feeds/feed_id
      organizations/organization_number/feeds/feed_id
  """
    name = _messages.StringField(1, required=True)
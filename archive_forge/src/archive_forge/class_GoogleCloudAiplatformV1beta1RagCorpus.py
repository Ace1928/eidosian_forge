from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RagCorpus(_messages.Message):
    """A RagCorpus is a RagFile container and a project can have multiple
  RagCorpora.

  Fields:
    createTime: Output only. Timestamp when this RagCorpus was created.
    description: Optional. The description of the RagCorpus.
    displayName: Required. The display name of the RagCorpus. The name can be
      up to 128 characters long and can consist of any UTF-8 characters.
    name: Output only. The resource name of the RagCorpus.
    updateTime: Output only. Timestamp when this RagCorpus was last updated.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    name = _messages.StringField(4)
    updateTime = _messages.StringField(5)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TagApiDeploymentRevisionRequest(_messages.Message):
    """Request message for TagApiDeploymentRevision.

  Fields:
    tag: Required. The tag to apply. The tag should be at most 40 characters,
      and match `a-z{3,39}`.
  """
    tag = _messages.StringField(1)
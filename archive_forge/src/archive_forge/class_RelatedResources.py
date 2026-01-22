from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RelatedResources(_messages.Message):
    """The related resources of the primary resource.

  Fields:
    relatedResources: The detailed related resources of the primary resource.
  """
    relatedResources = _messages.MessageField('RelatedResource', 1, repeated=True)
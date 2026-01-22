from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRoutersGetNamedSetRequest(_messages.Message):
    """A ComputeRoutersGetNamedSetRequest object.

  Fields:
    namedSet: The Named Set name for this request. Name must conform to
      RFC1035
    project: Project ID for this request.
    region: Name of the region for this request.
    router: Name of the Router resource to query for the named set. The name
      should conform to RFC1035.
  """
    namedSet = _messages.StringField(1)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)
    router = _messages.StringField(4, required=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRegionUrlMapsPatchRequest(_messages.Message):
    """A ComputeRegionUrlMapsPatchRequest object.

  Fields:
    project: Project ID for this request.
    region: Name of the region scoping this request.
    requestId: begin_interface: MixerMutationRequestBuilder Request ID to
      support idempotency.
    urlMap: Name of the UrlMap resource to patch.
    urlMapResource: A UrlMap resource to be passed as the request body.
  """
    project = _messages.StringField(1, required=True)
    region = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    urlMap = _messages.StringField(4, required=True)
    urlMapResource = _messages.MessageField('UrlMap', 5)
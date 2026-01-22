from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManifestsListResponse(_messages.Message):
    """A response containing a partial list of manifests and a page token used
  to build the next request if the request has been truncated.

  Fields:
    manifests: Output only. Manifests contained in this list response.
    nextPageToken: Output only. A token used to continue a truncated list
      request.
  """
    manifests = _messages.MessageField('Manifest', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
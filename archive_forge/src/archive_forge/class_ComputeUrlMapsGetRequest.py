from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeUrlMapsGetRequest(_messages.Message):
    """A ComputeUrlMapsGetRequest object.

  Fields:
    project: Project ID for this request.
    urlMap: Name of the UrlMap resource to return.
  """
    project = _messages.StringField(1, required=True)
    urlMap = _messages.StringField(2, required=True)
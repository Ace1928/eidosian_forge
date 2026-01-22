from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalStandaloneClustersListRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalStandaloneClustersListRequest
  object.

  Fields:
    pageSize: Requested page size. Server may return fewer items than
      requested. If unspecified, at most 50 clusters will be returned. The
      maximum value is 1000; values above 1000 will be coerced to 1000.
    pageToken: A token identifying a page of results the server should return.
    parent: Required. The parent of the project and location where the
      clusters are listed in. Format:
      "projects/{project}/locations/{location}"
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
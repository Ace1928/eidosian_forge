from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsWorkerPoolsListRequest(_messages.Message):
    """A CloudbuildProjectsLocationsWorkerPoolsListRequest object.

  Fields:
    pageSize: The maximum number of `WorkerPool`s to return. The service may
      return fewer than this value. If omitted, the server will use a sensible
      default.
    pageToken: A page token, received from a previous `ListWorkerPools` call.
      Provide this to retrieve the subsequent page.
    parent: Required. The parent of the collection of `WorkerPools`. Format:
      `projects/{project}/locations/{location}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
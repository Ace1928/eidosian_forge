from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TpuProjectsLocationsQueuedResourcesDeleteRequest(_messages.Message):
    """A TpuProjectsLocationsQueuedResourcesDeleteRequest object.

  Fields:
    force: Optional. If set to true, all running nodes belonging to this
      queued resource will be deleted first and then the queued resource will
      be deleted. Otherwise (i.e. force=false), the queued resource will only
      be deleted if its nodes have already been deleted or the queued resource
      is in the ACCEPTED, FAILED, or SUSPENDED state.
    name: Required. The resource name.
    requestId: Optional. Idempotent request UUID.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
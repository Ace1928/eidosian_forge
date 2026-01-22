from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TpuProjectsLocationsNodesStopRequest(_messages.Message):
    """A TpuProjectsLocationsNodesStopRequest object.

  Fields:
    name: Required. The resource name.
    stopNodeRequest: A StopNodeRequest resource to be passed as the request
      body.
  """
    name = _messages.StringField(1, required=True)
    stopNodeRequest = _messages.MessageField('StopNodeRequest', 2)
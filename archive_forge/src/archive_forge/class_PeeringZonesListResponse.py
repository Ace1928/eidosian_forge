from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PeeringZonesListResponse(_messages.Message):
    """A PeeringZonesListResponse object.

  Fields:
    header: A ResponseHeader attribute.
    kind: A string attribute.
    nextPageToken: The presence of this field indicates that there exist more
      results following your last page of results in pagination order. To
      fetch them, make another list request using this value as your page
      token. This lets you the complete contents of even very large
      collections one page at a time. However, if the contents of the
      collection change between the first and last paginated list request, the
      set of all elements returned are an inconsistent view of the collection.
      You cannot retrieve a consistent snapshot of a collection larger than
      the maximum page size.
    peeringZones: The active peering zone resources matching the request. Each
      ManagedZone has only the id field set.
  """
    header = _messages.MessageField('ResponseHeader', 1)
    kind = _messages.StringField(2, default='dns#peeringZonesListResponse')
    nextPageToken = _messages.StringField(3)
    peeringZones = _messages.MessageField('ManagedZone', 4, repeated=True)
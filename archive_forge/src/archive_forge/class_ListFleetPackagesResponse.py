from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListFleetPackagesResponse(_messages.Message):
    """Message for response to listing FleetPackage

  Fields:
    fleetPackages: The list of FleetPackage
    nextPageToken: A token identifying a page of results the server should
      return.
    unreachable: Locations that could not be reached.
  """
    fleetPackages = _messages.MessageField('FleetPackage', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)
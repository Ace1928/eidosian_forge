from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocationPolicy(_messages.Message):
    """A LocationPolicy object.

  Fields:
    allowedLocations: A list of allowed location names represented by internal
      URLs. Each location can be a region or a zone. Only one region or
      multiple zones in one region is supported now. For example,
      ["regions/us-central1"] allow VMs in any zones in region us-central1.
      ["zones/us-central1-a", "zones/us-central1-c"] only allow VMs in zones
      us-central1-a and us-central1-c. Mixing locations from different regions
      would cause errors. For example, ["regions/us-central1", "zones/us-
      central1-a", "zones/us-central1-b", "zones/us-west1-a"] contains
      locations from two distinct regions: us-central1 and us-west1. This
      combination will trigger an error.
    deniedLocations: A list of denied location names. Not yet implemented.
  """
    allowedLocations = _messages.StringField(1, repeated=True)
    deniedLocations = _messages.StringField(2, repeated=True)
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ExtractZonesFromLocationsListResponse(response, args):
    """Extract the zones from a list of GCP locations."""
    del args
    for location in response:
        if IsZonal(location.locationId):
            yield location
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ExtractRegionsFromLocationsListResponse(response, args):
    """Extract the regions from a list of GCP locations."""
    del args
    for location in response:
        if IsRegional(location.locationId):
            yield location
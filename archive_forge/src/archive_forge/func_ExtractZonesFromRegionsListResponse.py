from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
def ExtractZonesFromRegionsListResponse(response, args):
    for region in response:
        if args.IsSpecified('region') and region.locationId != args.region:
            continue
        if not region.metadata:
            continue
        metadata = encoding.MessageToDict(region.metadata)
        for zone in metadata.get('availableZones', []):
            zone = RedisZone(name=zone, region=region.locationId)
            yield zone
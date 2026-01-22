from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
def _maybe_add_locations(_, args, req):
    if not args.storage_location_names:
        return req
    locations_msg = messages.SnapshotSettingsStorageLocationSettings.LocationsValue(additionalProperties=[_wrap_location_name(location, messages) for location in args.storage_location_names])
    _ensure_location_field(req, messages)
    req.snapshotSettings.storageLocation.locations = locations_msg
    return req
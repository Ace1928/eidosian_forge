from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
def AddSnapshotNameToRequest(ref, args, req):
    """Python hook for yaml commands to process the source snapshot name."""
    location = args.source_snapshot_region or ref.locationsId
    if args.source_snapshot is None or location is None:
        return req
    project = properties.VALUES.core.project.Get(required=True)
    req.restoreInstanceRequest.sourceSnapshot = SNAPSHOT_NAME_TEMPLATE.format(project, location, args.source_snapshot)
    return req
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _DeduceRegionInProject(resources, current_project, disk_resource, sample, region, region_name, replica_zones):
    """Deduce region from zones in given project."""
    current_zones = [resources.Parse(zone, collection='compute.zones', params={'project': current_project}) for zone in replica_zones]
    for zone in current_zones:
        if zone.project != current_project:
            raise exceptions.InvalidArgumentException('--zone', 'Zone [{}] lives in different project than disk [{}].'.format(six.text_type(zone.SelfLink()), six.text_type(disk_resource.SelfLink())))
    for i in range(len(current_zones) - 1):
        if utils.ZoneNameToRegionName(current_zones[i].zone) != utils.ZoneNameToRegionName(current_zones[i + 1].zone):
            raise exceptions.InvalidArgumentException('--replica-zones', 'Zones [{}, {}] live in different regions [{}, {}], but should live in the same.'.format(current_zones[i].zone, current_zones[i + 1].zone, utils.ZoneNameToRegionName(current_zones[i].zone), utils.ZoneNameToRegionName(current_zones[i + 1].zone)))
    result = utils.ZoneNameToRegionName(current_zones[0].zone)
    if region is not None and region_name != sample and (region_name != result):
        raise exceptions.InvalidArgumentException('--replica-zones', 'Region from [--replica-zones] ({}) is different from [--region] ({}).'.format(result, region_name))
    return result
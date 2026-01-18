import argparse
import collections
import copy
import os
from oslo_utils import strutils
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3 import availability_zones
def treeizeAvailabilityZone(zone):
    """Builds a tree view for availability zones."""
    AvailabilityZone = availability_zones.AvailabilityZone
    az = AvailabilityZone(zone.manager, copy.deepcopy(zone._info), zone._loaded)
    result = []
    az.zoneName = zone.zoneName
    az.zoneState = 'available' if zone.zoneState['available'] else 'not available'
    az._info['zoneName'] = az.zoneName
    az._info['zoneState'] = az.zoneState
    result.append(az)
    if getattr(zone, 'hosts', None) and zone.hosts is not None:
        for host, services in zone.hosts.items():
            az = AvailabilityZone(zone.manager, copy.deepcopy(zone._info), zone._loaded)
            az.zoneName = '|- %s' % host
            az.zoneState = ''
            az._info['zoneName'] = az.zoneName
            az._info['zoneState'] = az.zoneState
            result.append(az)
            for svc, state in services.items():
                az = AvailabilityZone(zone.manager, copy.deepcopy(zone._info), zone._loaded)
                az.zoneName = '| |- %s' % svc
                az.zoneState = '%s %s %s' % ('enabled' if state['active'] else 'disabled', ':-)' if state['available'] else 'XXX', state['updated_at'])
                az._info['zoneName'] = az.zoneName
                az._info['zoneState'] = az.zoneState
                result.append(az)
    return result
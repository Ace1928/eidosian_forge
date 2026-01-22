import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
def _print_absolute_limits(limits):
    """Prints absolute limits."""

    class Limit(object):

        def __init__(self, name, used, max, other):
            self.name = name
            self.used = used
            self.max = max
            self.other = other
    limit_map = {'maxServerMeta': {'name': 'Server Meta', 'type': 'max'}, 'maxPersonality': {'name': 'Personality', 'type': 'max'}, 'maxPersonalitySize': {'name': 'Personality Size', 'type': 'max'}, 'maxImageMeta': {'name': 'ImageMeta', 'type': 'max'}, 'maxTotalKeypairs': {'name': 'Keypairs', 'type': 'max'}, 'totalCoresUsed': {'name': 'Cores', 'type': 'used'}, 'maxTotalCores': {'name': 'Cores', 'type': 'max'}, 'totalRAMUsed': {'name': 'RAM', 'type': 'used'}, 'maxTotalRAMSize': {'name': 'RAM', 'type': 'max'}, 'totalInstancesUsed': {'name': 'Instances', 'type': 'used'}, 'maxTotalInstances': {'name': 'Instances', 'type': 'max'}, 'totalFloatingIpsUsed': {'name': 'FloatingIps', 'type': 'used'}, 'maxTotalFloatingIps': {'name': 'FloatingIps', 'type': 'max'}, 'totalSecurityGroupsUsed': {'name': 'SecurityGroups', 'type': 'used'}, 'maxSecurityGroups': {'name': 'SecurityGroups', 'type': 'max'}, 'maxSecurityGroupRules': {'name': 'SecurityGroupRules', 'type': 'max'}, 'maxServerGroups': {'name': 'ServerGroups', 'type': 'max'}, 'totalServerGroupsUsed': {'name': 'ServerGroups', 'type': 'used'}, 'maxServerGroupMembers': {'name': 'ServerGroupMembers', 'type': 'max'}}
    max = {}
    used = {}
    other = {}
    limit_names = []
    columns = ['Name', 'Used', 'Max']
    for limit in limits:
        map = limit_map.get(limit.name, {'name': limit.name, 'type': 'other'})
        name = map['name']
        if map['type'] == 'max':
            max[name] = limit.value
        elif map['type'] == 'used':
            used[name] = limit.value
        else:
            other[name] = limit.value
            if 'Other' not in columns:
                columns.append('Other')
        if name not in limit_names:
            limit_names.append(name)
    limit_names.sort()
    limit_list = []
    for name in limit_names:
        limit_list.append(Limit(name, used.get(name, '-'), max.get(name, '-'), other.get(name, '-')))
    utils.print_list(limit_list, columns)
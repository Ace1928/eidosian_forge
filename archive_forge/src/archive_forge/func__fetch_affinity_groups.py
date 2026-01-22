from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _fetch_affinity_groups(self, cluster_service):
    affinity_groups = set()
    affinity_groups_service = cluster_service.affinity_groups_service()
    for affinity_group in affinity_groups_service.list():
        affinity_groups.add(affinity_group.name)
    return list(affinity_groups)
from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _is_compatible_versions(self, primary_conn, second_conn, cluster_mapping):
    """ Validate cluster versions """
    service_primary = primary_conn.system_service().clusters_service()
    service_sec = second_conn.system_service().clusters_service()
    for cluster_map in cluster_mapping:
        search_prime = 'name=%s' % cluster_map['primary_name']
        search_sec = 'name=%s' % cluster_map['secondary_name']
        cluster_prime = service_primary.list(search=search_prime)[0]
        cluster_sec = service_sec.list(search=search_sec)[0]
        prime_ver = cluster_prime.version
        sec_ver = cluster_sec.version
        if prime_ver.major != sec_ver.major or prime_ver.minor != sec_ver.minor:
            print("%s%sClusters have incompatible versions. primary setup ('%s' %s.%s) not equal to secondary setup ('%s' %s.%s)%s" % (FAIL, PREFIX, cluster_prime.name, prime_ver.major, prime_ver.minor, cluster_sec.name, sec_ver.major, sec_ver.minor, END))
            return False
    return True
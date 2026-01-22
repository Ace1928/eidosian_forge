from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _validate_duplicate_keys(self, var_file):
    clusters = 'clusters'
    domains = 'domains'
    roles = 'roles'
    aff_groups = 'aff_groups'
    aff_labels = 'aff_labels'
    network = 'network'
    key1 = 'primary_name'
    key2 = 'secondary_name'
    dr_primary_name = 'dr_primary_name'
    dr_secondary_name = 'dr_secondary_name'
    duplicates = self._get_dups(var_file, [[clusters, self.cluster_map, key1, key2], [domains, self.domain_map, dr_primary_name, dr_secondary_name], [roles, self.role_map, key1, key2], [aff_groups, self.aff_group_map, key1, key2], [aff_labels, self.aff_label_map, key1, key2]])
    duplicates[network] = self._get_dup_network(var_file)
    return not self._print_duplicate_keys(duplicates, [clusters, domains, roles, aff_groups, aff_labels, network])
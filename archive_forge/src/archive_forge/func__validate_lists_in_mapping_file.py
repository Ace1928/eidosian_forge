from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _validate_lists_in_mapping_file(self, mapping_vars):
    return self._is_list(mapping_vars, self.cluster_map) and self._is_list(mapping_vars, self.domain_map) and self._is_list(mapping_vars, self.role_map) and self._is_list(mapping_vars, self.aff_group_map) and self._is_list(mapping_vars, self.aff_label_map) and self._is_list(mapping_vars, self.network_map)
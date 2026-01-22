import argparse as argparse_mod
import collections
import copy
import errno
import json
import os
import re
import sys
import typing as ty
import warnings
from keystoneauth1 import adapter
from keystoneauth1 import loading
import platformdirs
import yaml
from openstack import _log
from openstack.config import _util
from openstack.config import cloud_region
from openstack.config import defaults
from openstack.config import vendors
from openstack import exceptions
from openstack import warnings as os_warnings
def _get_known_regions(self, cloud):
    config = _util.normalize_keys(self.cloud_config['clouds'][cloud])
    if 'regions' in config:
        return self._expand_regions(config['regions'])
    elif 'region_name' in config:
        if isinstance(config['region_name'], list):
            regions = config['region_name']
        else:
            regions = config['region_name'].split(',')
        if len(regions) > 1:
            warnings.warn(f'Comma separated lists in region_name are deprecated. Please use a yaml list in the regions parameter in {self.config_filename} instead.', os_warnings.OpenStackDeprecationWarning)
        return self._expand_regions(regions)
    else:
        new_cloud: ty.Dict[str, ty.Any] = {}
        our_cloud = self.cloud_config['clouds'].get(cloud, {})
        self._expand_vendor_profile(cloud, new_cloud, our_cloud)
        if 'regions' in new_cloud and new_cloud['regions']:
            return self._expand_regions(new_cloud['regions'])
        elif 'region_name' in new_cloud and new_cloud['region_name']:
            return [self._expand_region_name(new_cloud['region_name'])]
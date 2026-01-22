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
def _fix_backwards_project(self, cloud):
    mappings = {'domain_id': ('domain_id', 'domain-id'), 'domain_name': ('domain_name', 'domain-name'), 'user_domain_id': ('user_domain_id', 'user-domain-id'), 'user_domain_name': ('user_domain_name', 'user-domain-name'), 'project_domain_id': ('project_domain_id', 'project-domain-id'), 'project_domain_name': ('project_domain_name', 'project-domain-name'), 'token': ('auth-token', 'auth_token', 'token')}
    if cloud.get('auth_type', None) == 'v2password':
        mappings['tenant_id'] = ('project_id', 'project-id', 'tenant_id', 'tenant-id')
        mappings['tenant_name'] = ('project_name', 'project-name', 'tenant_name', 'tenant-name')
    else:
        mappings['project_id'] = ('tenant_id', 'tenant-id', 'project_id', 'project-id')
        mappings['project_name'] = ('tenant_name', 'tenant-name', 'project_name', 'project-name')
    for target_key, possible_values in mappings.items():
        target = None
        for key in possible_values:
            if key in cloud['auth']:
                target = str(cloud['auth'][key])
                del cloud['auth'][key]
            if key in cloud:
                target = str(cloud[key])
                del cloud[key]
        if target:
            cloud['auth'][target_key] = target
    return cloud
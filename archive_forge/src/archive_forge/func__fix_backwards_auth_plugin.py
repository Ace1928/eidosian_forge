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
def _fix_backwards_auth_plugin(self, cloud):
    mappings = {'auth_type': ('auth_plugin', 'auth_type')}
    for target_key, possible_values in mappings.items():
        target = None
        for key in possible_values:
            if key in cloud:
                target = cloud[key]
                del cloud[key]
        cloud[target_key] = target
    return cloud
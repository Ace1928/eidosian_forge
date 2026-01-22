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
def _get_auth_loader(self, config):
    if config['auth_type'] in (None, 'None', ''):
        config['auth_type'] = 'none'
    elif config['auth_type'] == 'token_endpoint':
        config['auth_type'] = 'admin_token'
    loader = loading.get_plugin_loader(config['auth_type'])
    if config['auth_type'] == 'v3multifactor':
        loader._methods = config.get('auth_methods')
    return loader
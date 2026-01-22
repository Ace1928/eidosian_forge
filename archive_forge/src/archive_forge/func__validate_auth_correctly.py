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
def _validate_auth_correctly(self, config, loader):
    plugin_options = loader.get_options()
    for p_opt in plugin_options:
        winning_value = self._find_winning_auth_value(p_opt, config)
        if not winning_value:
            winning_value = self._find_winning_auth_value(p_opt, config['auth'])
        config = self._clean_up_after_ourselves(config, p_opt, winning_value)
        config = self.option_prompt(config, p_opt)
    return config
from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def _merge_auth_option(config, module, module_option_name, env_var_name, config_attr_name):
    """Merge the values for an authentication attribute from ansible module options and
    environment variables with the values specified in a configuration file"""
    _debug('Merging {0}'.format(module_option_name))
    auth_attribute = module.params.get(module_option_name)
    _debug('\t Ansible module option {0} = {1}'.format(module_option_name, auth_attribute))
    if not auth_attribute:
        if env_var_name in os.environ:
            auth_attribute = os.environ[env_var_name]
            _debug('\t Environment variable {0} = {1}'.format(env_var_name, auth_attribute))
    if auth_attribute:
        _debug('Updating config attribute {0} -> {1} '.format(config_attr_name, auth_attribute))
        config.update({config_attr_name: auth_attribute})
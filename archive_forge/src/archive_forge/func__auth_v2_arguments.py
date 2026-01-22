import logging
from openstack.config import exceptions as sdk_exceptions
from openstack.config import loader as config
from oslo_utils import strutils
def _auth_v2_arguments(self, config):
    """Set up v2-required arguments from v3 info

        Migrated from auth.build_auth_params()
        """
    if 'auth_type' in config and config['auth_type'].startswith('v2'):
        if 'project_id' in config['auth']:
            config['auth']['tenant_id'] = config['auth']['project_id']
        if 'project_name' in config['auth']:
            config['auth']['tenant_name'] = config['auth']['project_name']
    return config
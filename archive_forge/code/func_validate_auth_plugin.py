import json
from keystoneauth1 import loading as ks_loading
from oslo_log import log as logging
from heat.common import exception
def validate_auth_plugin(auth_plugin, keystone_session):
    """Validate if this auth_plugin is valid to use."""
    try:
        auth_plugin.get_token(keystone_session)
    except Exception as e:
        failure_reason = 'Failed to validate auth_plugin with error %s. Please make sure the credential you provide is correct. Also make sure the it is a valid Keystone auth plugin type and contain in your environment.' % e
        raise exception.AuthorizationFailure(failure_reason=failure_reason)
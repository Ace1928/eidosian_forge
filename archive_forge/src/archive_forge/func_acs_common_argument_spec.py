from __future__ import (absolute_import, division, print_function)
import os
import json
import traceback
from ansible.module_utils.basic import env_fallback
def acs_common_argument_spec():
    return dict(alicloud_access_key=dict(aliases=['access_key_id', 'access_key'], no_log=True, fallback=(env_fallback, ['ALICLOUD_ACCESS_KEY', 'ALICLOUD_ACCESS_KEY_ID'])), alicloud_secret_key=dict(aliases=['secret_access_key', 'secret_key'], no_log=True, fallback=(env_fallback, ['ALICLOUD_SECRET_KEY', 'ALICLOUD_SECRET_ACCESS_KEY'])), alicloud_security_token=dict(aliases=['security_token'], no_log=True, fallback=(env_fallback, ['ALICLOUD_SECURITY_TOKEN'])), ecs_role_name=dict(aliases=['role_name'], fallback=(env_fallback, ['ALICLOUD_ECS_ROLE_NAME'])))
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.auth import external_account_authorized_user
from googlecloudsdk.api_lib.auth import util as auth_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
def _MakeThirdPartyClientConfig(login_config_data, is_adc):
    client_id = config.CLOUDSDK_CLIENT_ID
    client_secret = config.CLOUDSDK_CLIENT_NOTSOSECRET
    return {'installed': {'client_id': client_id, 'client_secret': client_secret, 'auth_uri': login_config_data['auth_url'], 'token_uri': login_config_data['token_url'], 'token_info_url': login_config_data['token_info_url'], 'audience': login_config_data['audience'], '3pi': True, 'is_adc': is_adc}}
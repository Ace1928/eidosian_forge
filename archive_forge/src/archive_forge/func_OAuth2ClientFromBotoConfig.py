from __future__ import absolute_import
import io
import json
import os
import sys
import time
import webbrowser
from gcs_oauth2_boto_plugin import oauth2_client
import oauth2client.client
from six.moves import input  # pylint: disable=redefined-builtin
def OAuth2ClientFromBotoConfig(config, cred_type=oauth2_client.CredTypes.OAUTH2_USER_ACCOUNT):
    """Create a client type based on credentials supplied in boto config."""
    token_cache = None
    token_cache_type = config.get('OAuth2', 'token_cache', 'file_system')
    if token_cache_type == 'file_system':
        if config.has_option('OAuth2', 'token_cache_path_pattern'):
            token_cache = oauth2_client.FileSystemTokenCache(path_pattern=config.get('OAuth2', 'token_cache_path_pattern'))
        else:
            token_cache = oauth2_client.FileSystemTokenCache()
    elif token_cache_type == 'in_memory':
        token_cache = oauth2_client.InMemoryTokenCache()
    else:
        raise Exception('Invalid value for config option OAuth2/token_cache: %s' % token_cache_type)
    proxy_host = None
    proxy_port = None
    proxy_user = None
    proxy_pass = None
    if config.has_option('Boto', 'proxy') and config.has_option('Boto', 'proxy_port'):
        proxy_host = config.get('Boto', 'proxy')
        proxy_port = int(config.get('Boto', 'proxy_port'))
        proxy_user = config.get('Boto', 'proxy_user', None)
        proxy_pass = config.get('Boto', 'proxy_pass', None)
    provider_authorization_uri = config.get('OAuth2', 'provider_authorization_uri', GOOGLE_OAUTH2_PROVIDER_AUTHORIZATION_URI)
    provider_token_uri = config.get('OAuth2', 'provider_token_uri', GOOGLE_OAUTH2_PROVIDER_TOKEN_URI)
    if cred_type == oauth2_client.CredTypes.OAUTH2_SERVICE_ACCOUNT:
        service_client_id = config.get('Credentials', 'gs_service_client_id', '')
        private_key_filename = config.get('Credentials', 'gs_service_key_file', '')
        with io.open(private_key_filename, 'rb') as private_key_file:
            private_key = private_key_file.read()
        keyfile_is_utf8 = False
        try:
            private_key = private_key.decode(UTF8)
            keyfile_is_utf8 = True
        except UnicodeDecodeError:
            pass
        if keyfile_is_utf8:
            try:
                json_key_dict = json.loads(private_key)
            except ValueError:
                raise Exception('Could not parse JSON keyfile "%s" as valid JSON' % private_key_filename)
            for json_entry in ('client_id', 'client_email', 'private_key_id', 'private_key'):
                if json_entry not in json_key_dict:
                    raise Exception('The JSON private key file at %s did not contain the required entry: %s' % (private_key_filename, json_entry))
            return oauth2_client.OAuth2JsonServiceAccountClient(json_key_dict, access_token_cache=token_cache, auth_uri=provider_authorization_uri, token_uri=provider_token_uri, disable_ssl_certificate_validation=not config.getbool('Boto', 'https_validate_certificates', True), proxy_host=proxy_host, proxy_port=proxy_port, proxy_user=proxy_user, proxy_pass=proxy_pass)
        else:
            key_file_pass = config.get('Credentials', 'gs_service_key_file_password', GOOGLE_OAUTH2_DEFAULT_FILE_PASSWORD)
            return oauth2_client.OAuth2ServiceAccountClient(service_client_id, private_key, key_file_pass, access_token_cache=token_cache, auth_uri=provider_authorization_uri, token_uri=provider_token_uri, disable_ssl_certificate_validation=not config.getbool('Boto', 'https_validate_certificates', True), proxy_host=proxy_host, proxy_port=proxy_port, proxy_user=proxy_user, proxy_pass=proxy_pass)
    elif cred_type == oauth2_client.CredTypes.OAUTH2_USER_ACCOUNT:
        client_id = config.get('OAuth2', 'client_id', os.environ.get('OAUTH2_CLIENT_ID', CLIENT_ID))
        if not client_id:
            raise Exception('client_id for your application obtained from https://console.developers.google.com must be set in a boto config or with OAUTH2_CLIENT_ID environment variable or with gcs_oauth2_boto_plugin.SetFallbackClientIdAndSecret function.')
        client_secret = config.get('OAuth2', 'client_secret', os.environ.get('OAUTH2_CLIENT_SECRET', CLIENT_SECRET))
        ca_certs_file = config.get_value('Boto', 'ca_certificates_file')
        if ca_certs_file == 'system':
            ca_certs_file = None
        if not client_secret:
            raise Exception('client_secret for your application obtained from https://console.developers.google.com must be set in a boto config or with OAUTH2_CLIENT_SECRET environment variable or with gcs_oauth2_boto_plugin.SetFallbackClientIdAndSecret function.')
        return oauth2_client.OAuth2UserAccountClient(provider_token_uri, client_id, client_secret, config.get('Credentials', 'gs_oauth2_refresh_token'), auth_uri=provider_authorization_uri, access_token_cache=token_cache, disable_ssl_certificate_validation=not config.getbool('Boto', 'https_validate_certificates', True), proxy_host=proxy_host, proxy_port=proxy_port, proxy_user=proxy_user, proxy_pass=proxy_pass, ca_certs_file=ca_certs_file)
    else:
        raise Exception('You have attempted to create an OAuth2 client without setting up OAuth2 credentials.')
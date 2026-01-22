from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import datetime
import json
import multiprocessing
import os
import signal
import socket
import stat
import sys
import textwrap
import time
import webbrowser
from six.moves import input
from six.moves.http_client import ResponseNotReady
import boto
from boto.provider import Provider
import gslib
from gslib.command import Command
from gslib.command import DEFAULT_TASK_ESTIMATION_THRESHOLD
from gslib.commands.compose import MAX_COMPOSE_ARITY
from gslib.cred_types import CredTypes
from gslib.exception import AbortException
from gslib.exception import CommandException
from gslib.metrics import CheckAndMaybePromptForAnalyticsEnabling
from gslib.sig_handling import RegisterSignalHandler
from gslib.utils import constants
from gslib.utils import system_util
from gslib.utils.hashing_helper import CHECK_HASH_ALWAYS
from gslib.utils.hashing_helper import CHECK_HASH_IF_FAST_ELSE_FAIL
from gslib.utils.hashing_helper import CHECK_HASH_IF_FAST_ELSE_SKIP
from gslib.utils.hashing_helper import CHECK_HASH_NEVER
from gslib.utils.parallelism_framework_util import ShouldProhibitMultiprocessing
from httplib2 import ServerNotFoundError
from oauth2client.client import HAS_CRYPTO
def _WriteBotoConfigFile(self, config_file, cred_type=CredTypes.OAUTH2_USER_ACCOUNT, configure_auth=True):
    """Creates a boto config file interactively.

    Needed credentials are obtained interactively, either by asking the user for
    access key and secret, or by walking the user through the OAuth2 approval
    flow.

    Args:
      config_file: File object to which the resulting config file will be
          written.
      cred_type: There are three options:
        - for HMAC, ask the user for access key and secret
        - for OAUTH2_USER_ACCOUNT, raise an error
        - for OAUTH2_SERVICE_ACCOUNT, prompt the user for OAuth2 for service
          account email address and private key file (and if the file is a .p12
          file, the password for that file).
      configure_auth: Boolean, whether or not to configure authentication in
          the generated file.
    """
    provider_map = {'aws': 'aws', 'google': 'gs'}
    uri_map = {'aws': 's3', 'google': 'gs'}
    key_ids = {}
    sec_keys = {}
    service_account_key_is_json = False
    if configure_auth:
        if cred_type == CredTypes.OAUTH2_SERVICE_ACCOUNT:
            gs_service_key_file = input('What is the full path to your private key file? ')
            try:
                with open(gs_service_key_file, 'rb') as key_file_fp:
                    json.loads(key_file_fp.read())
                service_account_key_is_json = True
            except ValueError:
                if not HAS_CRYPTO:
                    raise CommandException('Service account authentication via a .p12 file requires either\nPyOpenSSL or PyCrypto 2.6 or later. Please install either of these\nto proceed, use a JSON-format key file, or configure a different type of credentials.')
            if not service_account_key_is_json:
                gs_service_client_id = input('What is your service account email address? ')
                gs_service_key_file_password = input('\n'.join(textwrap.wrap("What is the password for your service key file [if you haven't set one explicitly, leave this line blank]?")) + ' ')
            self._CheckPrivateKeyFilePermissions(gs_service_key_file)
        elif cred_type == CredTypes.OAUTH2_USER_ACCOUNT:
            raise CommandException('The user account authentication flow no longer works as of February 1, 2023. Tokens generated before this date will continue to work. To authenticate with your user account, install gsutil via Cloud SDK and run "gcloud auth login"')
        elif cred_type == CredTypes.HMAC:
            got_creds = False
            for provider in provider_map:
                if provider == 'google':
                    key_ids[provider] = input('What is your %s access key ID? ' % provider)
                    sec_keys[provider] = input('What is your %s secret access key? ' % provider)
                    got_creds = True
                    if not key_ids[provider] or not sec_keys[provider]:
                        raise CommandException('Incomplete credentials provided. Please try again.')
            if not got_creds:
                raise CommandException('No credentials provided. Please try again.')
    config_file.write(CONFIG_PRELUDE_CONTENT.lstrip())
    config_file.write('# This file was created by gsutil version %s at %s.\n' % (gslib.VERSION, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    config_file.write('#\n# You can create additional configuration files by running\n# gsutil config [options] [-o <config-file>]\n\n\n')
    config_file.write('[Credentials]\n\n')
    if configure_auth:
        if cred_type == CredTypes.OAUTH2_SERVICE_ACCOUNT:
            config_file.write('# Google OAuth2 service account credentials (for "gs://" URIs):\n')
            config_file.write('gs_service_key_file = %s\n' % gs_service_key_file)
            if not service_account_key_is_json:
                config_file.write('gs_service_client_id = %s\n' % gs_service_client_id)
                if not gs_service_key_file_password:
                    config_file.write('# If you would like to set your password, you can do so\n# using the following commands (replaced with your\n# information):\n# "openssl pkcs12 -in cert1.p12 -out temp_cert.pem"\n# "openssl pkcs12 -export -in temp_cert.pem -out cert2.p12"\n# "rm -f temp_cert.pem"\n# Your initial password is "notasecret" - for more\n# information, please see \n# http://www.openssl.org/docs/apps/pkcs12.html.\n')
                    config_file.write('#gs_service_key_file_password =\n\n')
                else:
                    config_file.write('gs_service_key_file_password = %s\n\n' % gs_service_key_file_password)
        else:
            config_file.write('# To add Google OAuth2 credentials ("gs://" URIs), edit and uncomment the\n# following line:\n#gs_oauth2_refresh_token = <your OAuth2 refresh token>\n\n')
    elif system_util.InvokedViaCloudSdk():
        config_file.write('# Google OAuth2 credentials are managed by the Cloud SDK and\n# do not need to be present in this file.\n')
    for provider in provider_map:
        key_prefix = provider_map[provider]
        uri_scheme = uri_map[provider]
        if provider in key_ids and provider in sec_keys:
            config_file.write('# %s credentials ("%s://" URIs):\n' % (provider, uri_scheme))
            config_file.write('%s_access_key_id = %s\n' % (key_prefix, key_ids[provider]))
            config_file.write('%s_secret_access_key = %s\n' % (key_prefix, sec_keys[provider]))
        else:
            config_file.write('# To add HMAC %s credentials for "%s://" URIs, edit and uncomment the\n# following two lines:\n#%s_access_key_id = <your %s access key ID>\n#%s_secret_access_key = <your %s secret access key>\n' % (provider, uri_scheme, key_prefix, provider, key_prefix, provider))
        host_key = Provider.HostKeyMap[provider]
        config_file.write('# The ability to specify an alternate storage host and port\n# is primarily for cloud storage service developers.\n# Setting a non-default gs_host only works if prefer_api=xml.\n#%s_host = <alternate storage host address>\n#%s_port = <alternate storage host port>\n# In some cases, (e.g. VPC requests) the "host" HTTP header should\n# be different than the host used in the request URL.\n#%s_host_header = <alternate storage host header>\n' % (host_key, host_key, host_key))
        if host_key == 'gs':
            config_file.write('#%s_json_host = <alternate JSON API storage host address>\n#%s_json_port = <alternate JSON API storage host port>\n#%s_json_host_header = <alternate JSON API storage host header>\n\n' % (host_key, host_key, host_key))
            config_file.write('# To impersonate a service account for "%s://" URIs over\n# JSON API, edit and uncomment the following line:\n#%s_impersonate_service_account = <service account email>\n\n')
    config_file.write(textwrap.dedent('        # This configuration setting enables or disables mutual TLS\n        # authentication. The default value for this setting is "false". When\n        # set to "true", gsutil uses the configured client certificate as\n        # transport credential to access the APIs. The use of mTLS ensures that\n        # the access originates from a trusted enterprise device. When enabled,\n        # the client certificate is auto discovered using the endpoint\n        # verification agent. When set to "true" but no client certificate or\n        # key is found, users receive an error.\n        #use_client_certificate = False\n\n        # The command line to execute, which prints the\n        # certificate, private key, or password to use in\n        # conjunction with "use_client_certificate = True".\n        #cert_provider_command = <Absolute path to command to run for\n        #                         certification. Ex: "/scripts/gen_cert.sh">\n\n        '))
    config_file.write('%s\n' % CONFIG_BOTO_SECTION_CONTENT)
    self._WriteProxyConfigFileSection(config_file)
    config_file.write(CONFIG_GOOGLECOMPUTE_SECTION_CONTENT)
    config_file.write(CONFIG_INPUTLESS_GSUTIL_SECTION_CONTENT)
    config_file.write("\n# 'default_api_version' specifies the default Google Cloud Storage XML API\n# version to use. If not set below gsutil defaults to API version 1.\n")
    api_version = 2
    if cred_type == CredTypes.HMAC:
        api_version = 1
    config_file.write('default_api_version = %d\n' % api_version)
    if not system_util.InvokedViaCloudSdk():
        default_project_id = input('What is your project-id? ').strip()
        project_id_section_prelude = "\n# 'default_project_id' specifies the default Google Cloud Storage project ID to\n# use with the 'mb' and 'ls' commands. This default can be overridden by\n# specifying the -p option to the 'mb' and 'ls' commands.\n"
        if not default_project_id:
            raise CommandException('No default project ID entered. The default project ID is needed by the\nls and mb commands; please try again.')
        config_file.write('%sdefault_project_id = %s\n\n\n' % (project_id_section_prelude, default_project_id))
        CheckAndMaybePromptForAnalyticsEnabling()
    config_file.write(CONFIG_OAUTH2_CONFIG_CONTENT)
    config_file.write('#client_id = <OAuth2 client id>\n#client_secret = <OAuth2 client secret>\n')
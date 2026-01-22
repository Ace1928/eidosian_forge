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
class ConfigCommand(Command):
    """Implementation of gsutil config command."""
    command_spec = Command.CreateCommandSpec('config', command_name_aliases=['cfg', 'conf', 'configure'], usage_synopsis=_SYNOPSIS, min_args=0, max_args=0, supported_sub_args='abefhno:rs:w', supported_private_args=['reauth'], file_url_ok=False, provider_url_ok=False, urls_start_arg=0)
    help_spec = Command.HelpSpec(help_name='config', help_name_aliases=['cfg', 'conf', 'configure', 'aws', 's3'], help_type='command_help', help_one_line_summary='Obtain credentials and create configuration file', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})

    def _OpenConfigFile(self, file_path):
        """Creates and opens a configuration file for writing.

    The file is created with mode 0600, and attempts to open existing files will
    fail (the latter is important to prevent symlink attacks).

    It is the caller's responsibility to close the file.

    Args:
      file_path: Path of the file to be created.

    Returns:
      A writable file object for the opened file.

    Raises:
      CommandException: if an error occurred when opening the file (including
          when the file already exists).
    """
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, 'O_NOINHERIT'):
            flags |= os.O_NOINHERIT
        try:
            fd = os.open(file_path, flags, 384)
        except (OSError, IOError) as e:
            raise CommandException('Failed to open %s for writing: %s' % (file_path, e))
        return os.fdopen(fd, 'w')

    def _CheckPrivateKeyFilePermissions(self, file_path):
        """Checks that the file has reasonable permissions for a private key.

    In particular, check that the filename provided by the user is not
    world- or group-readable. If either of these are true, we issue a warning
    and offer to fix the permissions.

    Args:
      file_path: The name of the private key file.
    """
        if system_util.IS_WINDOWS:
            return
        st = os.stat(file_path)
        if bool((stat.S_IRGRP | stat.S_IROTH) & st.st_mode):
            self.logger.warn('\nYour private key file is readable by people other than yourself.\nThis is a security risk, since anyone with this information can use your service account.\n')
            fix_it = input('Would you like gsutil to change the file permissions for you? (y/N) ')
            if fix_it in ('y', 'Y'):
                try:
                    os.chmod(file_path, 256)
                    self.logger.info('\nThe permissions on your file have been successfully modified.\nThe only access allowed is readability by the user (permissions 0400 in chmod).')
                except Exception as _:
                    self.logger.warn('\nWe were unable to modify the permissions on your file.\nIf you would like to fix this yourself, consider running:\n"sudo chmod 400 </path/to/key>" for improved security.')
            else:
                self.logger.info('\nYou have chosen to allow this file to be readable by others.\nIf you would like to fix this yourself, consider running:\n"sudo chmod 400 </path/to/key>" for improved security.')

    def _WriteConfigLineMaybeCommented(self, config_file, name, value, desc):
        """Writes proxy name/value pair or comment line to config file.

    Writes proxy name/value pair if value is not None.  Otherwise writes
    comment line.

    Args:
      config_file: File object to which the resulting config file will be
          written.
      name: The config variable name.
      value: The value, or None.
      desc: Human readable description (for comment).
    """
        if not value:
            name = '#%s' % name
            value = '<%s>' % desc
        config_file.write('%s = %s\n' % (name, value))

    def _WriteProxyConfigFileSection(self, config_file):
        """Writes proxy section of configuration file.

    Args:
      config_file: File object to which the resulting config file will be
          written.
    """
        config = boto.config
        config_file.write('# To use a proxy, edit and uncomment the proxy and proxy_port lines.\n# If you need a user/password with this proxy, edit and uncomment\n# those lines as well. If your organization also disallows DNS\n# lookups by client machines, set proxy_rdns to True (the default).\n# If you have installed gsutil through the Cloud SDK and have \n# configured proxy settings in gcloud, those proxy settings will \n# override any other options (including those set here, along with \n# any settings in proxy-related environment variables). Otherwise, \n# if proxy_host and proxy_port are not specified in this file and\n# one of the OS environment variables http_proxy, https_proxy, or\n# HTTPS_PROXY is defined, gsutil will use the proxy server specified\n# in these environment variables, in order of precedence according\n# to how they are listed above.\n')
        self._WriteConfigLineMaybeCommented(config_file, 'proxy', config.get_value('Boto', 'proxy', None), 'proxy host')
        self._WriteConfigLineMaybeCommented(config_file, 'proxy_type', config.get_value('Boto', 'proxy_type', None), 'proxy type (socks4, socks5, http) | Defaults to http')
        self._WriteConfigLineMaybeCommented(config_file, 'proxy_port', config.get_value('Boto', 'proxy_port', None), 'proxy port')
        self._WriteConfigLineMaybeCommented(config_file, 'proxy_user', config.get_value('Boto', 'proxy_user', None), 'proxy user')
        self._WriteConfigLineMaybeCommented(config_file, 'proxy_pass', config.get_value('Boto', 'proxy_pass', None), 'proxy password')
        self._WriteConfigLineMaybeCommented(config_file, 'proxy_rdns', config.get_value('Boto', 'proxy_rdns', False), 'let proxy server perform DNS lookups (True,False); socks proxy not supported')

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

    def RunCommand(self):
        """Command entry point for the config command."""
        cred_type = CredTypes.OAUTH2_USER_ACCOUNT
        output_file_name = None
        has_a = False
        has_e = False
        configure_auth = True
        for opt, opt_arg in self.sub_opts:
            if opt == '-a':
                cred_type = CredTypes.HMAC
                has_a = True
            elif opt == '-e':
                cred_type = CredTypes.OAUTH2_SERVICE_ACCOUNT
                has_e = True
            elif opt == '-n':
                configure_auth = False
            elif opt == '-o':
                output_file_name = opt_arg
            else:
                self.RaiseInvalidArgumentException()
        if has_e and has_a:
            raise CommandException('Both -a and -e cannot be specified. Please see "gsutil help config" for more information.')
        if not configure_auth and (has_a or has_e):
            raise CommandException('The -a and -e flags cannot be specified with the -n flag. Please see "gsutil help config" for more information.')
        if system_util.InvokedViaCloudSdk() and system_util.CloudSdkCredPassingEnabled() and (not has_a) and configure_auth:
            raise CommandException('\n'.join(['OAuth2 is the preferred authentication mechanism with the Cloud SDK.', 'Run "gcloud auth login" to configure authentication, unless:', '\n'.join(textwrap.wrap('You don\'t want gsutil to use OAuth2 credentials from the Cloud SDK, but instead want to manage credentials with .boto files generated by running "gsutil config"; in which case run "gcloud config set pass_credentials_to_gsutil false".', initial_indent='- ', subsequent_indent='  ')), '\n'.join(textwrap.wrap('You want to authenticate with an HMAC access key and secret, in which case run "gsutil config -a".', initial_indent='- ', subsequent_indent='  '))]))
        if system_util.InvokedViaCloudSdk() and has_a:
            sys.stderr.write('\n'.join(textwrap.wrap('This command will configure HMAC credentials, but gsutil will use OAuth2 credentials from the Cloud SDK by default. To make sure the HMAC credentials are used, run: "gcloud config set pass_credentials_to_gsutil false".')) + '\n\n')
        default_config_path_bak = None
        if not output_file_name:
            boto_config_from_env = os.environ.get('BOTO_CONFIG', None)
            if boto_config_from_env:
                default_config_path = boto_config_from_env
            else:
                default_config_path = os.path.expanduser(os.path.join('~', '.boto'))
            if not os.path.exists(default_config_path):
                output_file_name = default_config_path
            else:
                default_config_path_bak = default_config_path + '.bak'
                if os.path.exists(default_config_path_bak):
                    raise CommandException('Cannot back up existing config file "%s": backup file exists ("%s").' % (default_config_path, default_config_path_bak))
                else:
                    try:
                        sys.stderr.write('Backing up existing config file "%s" to "%s"...\n' % (default_config_path, default_config_path_bak))
                        os.rename(default_config_path, default_config_path_bak)
                    except Exception as e:
                        raise CommandException('Failed to back up existing config file ("%s" -> "%s"): %s.' % (default_config_path, default_config_path_bak, e))
                    output_file_name = default_config_path
        if output_file_name == '-':
            output_file = sys.stdout
        else:
            output_file = self._OpenConfigFile(output_file_name)
            sys.stderr.write('\n'.join(textwrap.wrap('This command will create a boto config file at %s containing your credentials, based on your responses to the following questions.' % output_file_name)) + '\n')
        RegisterSignalHandler(signal.SIGINT, _CleanupHandler)
        try:
            self._WriteBotoConfigFile(output_file, cred_type=cred_type, configure_auth=configure_auth)
        except Exception as e:
            user_aborted = isinstance(e, AbortException)
            if user_aborted:
                sys.stderr.write('\nCaught ^C; cleaning up\n')
            if output_file_name != '-':
                output_file.close()
                os.unlink(output_file_name)
                try:
                    if default_config_path_bak:
                        sys.stderr.write('Restoring previous backed up file (%s)\n' % default_config_path_bak)
                        os.rename(default_config_path_bak, output_file_name)
                except Exception as e:
                    raise e
            raise
        if output_file_name != '-':
            output_file.close()
            if not boto.config.has_option('Boto', 'proxy'):
                sys.stderr.write('\n' + '\n'.join(textwrap.wrap('Boto config file "%s" created.\nIf you need to use a proxy to access the Internet please see the instructions in that file.' % output_file_name)) + '\n')
        return 0
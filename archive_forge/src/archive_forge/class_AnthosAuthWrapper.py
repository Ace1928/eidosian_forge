from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import copy
import json
import os
from googlecloudsdk.command_lib.anthos import flags
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.anthos.common import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
from six.moves import urllib
class AnthosAuthWrapper(binary_operations.StreamingBinaryBackedOperation):
    """Binary operation wrapper for anthoscli commands."""

    def __init__(self, **kwargs):
        custom_errors = {'MISSING_EXEC': messages.MISSING_AUTH_BINARY.format(binary='kubectl-anthos')}
        super(AnthosAuthWrapper, self).__init__(binary='kubectl-anthos', custom_errors=custom_errors, **kwargs)

    @property
    def default_config_path(self):
        return files.ExpandHomeAndVars(DEFAULT_LOGIN_CONFIG_PATH[platforms.OperatingSystem.Current().id])

    def _ParseLoginArgs(self, cluster, kube_config=None, login_config=None, login_config_cert=None, user=None, ldap_user=None, ldap_pass=None, dry_run=None, preferred_auth=None, server_url=None, **kwargs):
        del kwargs
        exec_args = ['login']
        if cluster:
            exec_args.extend(['--cluster', cluster])
        if kube_config:
            exec_args.extend(['--kubeconfig', kube_config])
        if login_config:
            exec_args.extend(['--login-config', login_config])
        if login_config_cert:
            exec_args.extend(['--login-config-cert', login_config_cert])
        if user:
            exec_args.extend(['--user', user])
        if dry_run:
            exec_args.extend(['--dry-run'])
        if ldap_pass and ldap_user:
            exec_args.extend(['--ldap-username', ldap_user, '--ldap-password', ldap_pass])
        if preferred_auth:
            exec_args.extend(['--preferred-auth', preferred_auth])
        if server_url:
            exec_args.extend(['--server', server_url])
        return exec_args

    def _ParseCreateLoginConfigArgs(self, kube_config, output_file=None, merge_from=None, **kwargs):
        del kwargs
        exec_args = ['create-login-config']
        exec_args.extend(['--kubeconfig', kube_config])
        if output_file:
            exec_args.extend(['--output', output_file])
        if merge_from:
            exec_args.extend(['--merge-from', merge_from])
        return exec_args

    def _ParseTokenArgs(self, token_type, cluster, aws_sts_region, id_token, access_token, access_token_expiry, refresh_token, client_id, client_secret, idp_certificate_authority_data, idp_issuer_url, kubeconfig_path, user, **kwargs):
        del kwargs
        exec_args = ['token']
        if token_type:
            exec_args.extend(['--type', token_type])
        if cluster:
            exec_args.extend(['--cluster', cluster])
        if aws_sts_region:
            exec_args.extend(['--aws-sts-region', aws_sts_region])
        if id_token:
            exec_args.extend(['--id-token', id_token])
        if access_token:
            exec_args.extend(['--access-token', access_token])
        if access_token_expiry:
            exec_args.extend(['--access-token-expiry', access_token_expiry])
        if refresh_token:
            exec_args.extend(['--refresh-token', refresh_token])
        if client_id:
            exec_args.extend(['--client-id', client_id])
        if client_secret:
            exec_args.extend(['--client-secret', client_secret])
        if idp_certificate_authority_data:
            exec_args.extend(['--idp-certificate-authority-data', idp_certificate_authority_data])
        if idp_issuer_url:
            exec_args.extend(['--idp-issuer-url', idp_issuer_url])
        if kubeconfig_path:
            exec_args.extend(['--kubeconfig-path', kubeconfig_path])
        if user:
            exec_args.extend(['--user', user])
        return exec_args

    def _ParseArgsForCommand(self, command, **kwargs):
        if command == 'login':
            return self._ParseLoginArgs(**kwargs)
        elif command == 'create-login-config':
            return self._ParseCreateLoginConfigArgs(**kwargs)
        elif command == 'version':
            return ['version']
        elif command == 'token':
            return self._ParseTokenArgs(**kwargs)
        else:
            raise binary_operations.InvalidOperationForBinary('Invalid Operation [{}] for kubectl-anthos'.format(command))
from __future__ import absolute_import, division, print_function
import copy
import os
import re
import datetime
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import exec_command
from ansible.module_utils.six import iteritems
from ansible.module_utils.parsing.convert_bool import (
from collections import defaultdict
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from .constants import (
class F5BaseClient(object):

    def __init__(self, *args, **kwargs):
        self.params = kwargs
        self.module = kwargs.get('module', None)
        load_params(self.params)
        self._client = None

    @property
    def api(self):
        raise F5ModuleError('Management root must be used from the concrete product classes.')

    def reconnect(self):
        """Attempts to reconnect to a device

        The existing token from a ManagementRoot can become invalid if you,
        for example, upgrade the device (such as is done in the *_software
        module.

        This method can be used to reconnect to a remote device without
        having to re-instantiate the ArgumentSpec and AnsibleF5Client classes
        it will use the same values that were initially provided to those
        classes

        :return:
        :raises iControlUnexpectedHTTPError
        """
        self._client = None

    @staticmethod
    def validate_params(key, store):
        if key in store and store[key] is not None:
            return True
        else:
            return False

    def merge_provider_params(self):
        result = dict()
        provider = self.params.get('provider', None)
        if not provider:
            provider = {}
        self.merge_provider_server_param(result, provider)
        self.merge_provider_server_port_param(result, provider)
        self.merge_provider_validate_certs_param(result, provider)
        self.merge_provider_auth_provider_param(result, provider)
        self.merge_provider_user_param(result, provider)
        self.merge_provider_timeout_param(result, provider)
        self.merge_provider_password_param(result, provider)
        self.merge_provider_no_f5_teem_param(result, provider)
        return result

    def merge_provider_server_param(self, result, provider):
        if self.validate_params('server', provider):
            result['server'] = provider['server']
        elif self.validate_params('F5_SERVER', os.environ):
            result['server'] = os.environ['F5_SERVER']
        else:
            raise F5ModuleError('Server parameter cannot be None or missing, please provide a valid value')

    def merge_provider_server_port_param(self, result, provider):
        if self.validate_params('server_port', provider):
            result['server_port'] = provider['server_port']
        elif self.validate_params('F5_SERVER_PORT', os.environ):
            result['server_port'] = os.environ['F5_SERVER_PORT']
        else:
            result['server_port'] = 443

    def merge_provider_validate_certs_param(self, result, provider):
        if self.validate_params('validate_certs', provider):
            result['validate_certs'] = provider['validate_certs']
        elif self.validate_params('F5_VALIDATE_CERTS', os.environ):
            result['validate_certs'] = os.environ['F5_VALIDATE_CERTS']
        else:
            result['validate_certs'] = True
        if result['validate_certs'] in BOOLEANS_TRUE:
            result['validate_certs'] = True
        else:
            result['validate_certs'] = False

    def merge_provider_auth_provider_param(self, result, provider):
        if self.validate_params('auth_provider', provider):
            result['auth_provider'] = provider['auth_provider']
        elif self.validate_params('F5_AUTH_PROVIDER', os.environ):
            result['auth_provider'] = os.environ['F5_AUTH_PROVIDER']
        else:
            result['auth_provider'] = None
        if result['auth_provider'] is not None and '__omit_place_holder__' in result['auth_provider']:
            result['auth_provider'] = None

    def merge_provider_user_param(self, result, provider):
        if self.validate_params('user', provider):
            result['user'] = provider['user']
        elif self.validate_params('F5_USER', os.environ):
            result['user'] = os.environ.get('F5_USER')
        elif self.validate_params('ANSIBLE_NET_USERNAME', os.environ):
            result['user'] = os.environ.get('ANSIBLE_NET_USERNAME')
        else:
            result['user'] = None

    def merge_provider_timeout_param(self, result, provider):
        if self.validate_params('timeout', provider):
            result['timeout'] = provider['timeout']
        elif self.validate_params('F5_TIMEOUT', os.environ):
            result['timeout'] = os.environ.get('F5_TIMEOUT')
        else:
            result['timeout'] = None

    def merge_provider_password_param(self, result, provider):
        if self.validate_params('password', provider):
            result['password'] = provider['password']
        elif self.validate_params('F5_PASSWORD', os.environ):
            result['password'] = os.environ.get('F5_PASSWORD')
        elif self.validate_params('ANSIBLE_NET_PASSWORD', os.environ):
            result['password'] = os.environ.get('ANSIBLE_NET_PASSWORD')
        else:
            result['password'] = None

    def merge_provider_no_f5_teem_param(self, result, provider):
        if self.validate_params('no_f5_teem', provider):
            result['no_f5_teem'] = provider['no_f5_teem']
        elif self.validate_params('F5_TEEM', os.environ):
            result['no_f5_teem'] = os.environ['F5_TEEM']
        elif self.validate_params('F5_TELEMETRY_OFF', os.environ):
            result['no_f5_teem'] = os.environ['F5_TELEMETRY_OFF']
        else:
            result['no_f5_teem'] = False
        if result['no_f5_teem'] in BOOLEANS_TRUE:
            result['no_f5_teem'] = True
        else:
            result['no_f5_teem'] = False
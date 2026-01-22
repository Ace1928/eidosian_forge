from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.validation import (
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import HashiVaultOptionGroupBase
class HashiVaultConnectionOptions(HashiVaultOptionGroupBase):
    """HashiVault option group class for connection options"""
    OPTIONS = ['url', 'proxies', 'ca_cert', 'validate_certs', 'namespace', 'timeout', 'retries', 'retry_action']
    ARGSPEC = dict(url=dict(type='str', default=None), proxies=dict(type='raw'), ca_cert=dict(type='str', aliases=['cacert'], default=None), validate_certs=dict(type='bool'), namespace=dict(type='str', default=None), timeout=dict(type='int'), retries=dict(type='raw'), retry_action=dict(type='str', choices=['ignore', 'warn'], default='warn'))
    _LATE_BINDING_ENV_VAR_OPTIONS = {'url': dict(env=['VAULT_ADDR'], required=True), 'ca_cert': dict(env=['VAULT_CACERT']), 'namespace': dict(env=['VAULT_NAMESPACE'])}
    _RETRIES_DEFAULT_PARAMS = {'status_forcelist': [412, 500, 502, 503], 'allowed_methods' if HAS_RETRIES and hasattr(urllib3.util.Retry.DEFAULT, 'allowed_methods') else 'method_whitelist': None, 'backoff_factor': 0.3}

    def __init__(self, option_adapter, retry_callback_generator=None):
        super(HashiVaultConnectionOptions, self).__init__(option_adapter)
        self._retry_callback_generator = retry_callback_generator

    def get_hvac_connection_options(self):
        """returns kwargs to be used for constructing an hvac.Client"""

        def _filter(k, v):
            return v is not None and k not in ('validate_certs', 'ca_cert')
        hvopts = self._options.get_filtered_options(_filter, *self.OPTIONS)
        hvopts['verify'] = self._conopt_verify
        retry_action = hvopts.pop('retry_action')
        if 'retries' in hvopts:
            hvopts['session'] = self._get_custom_requests_session(new_callback=self._retry_callback_generator(retry_action), **hvopts.pop('retries'))
        return hvopts

    def process_connection_options(self):
        """executes special processing required for certain options"""
        self.process_late_binding_env_vars(self._LATE_BINDING_ENV_VAR_OPTIONS)
        self._boolean_or_cacert()
        self._process_option_proxies()
        self._process_option_retries()

    def _get_custom_requests_session(self, **retry_kwargs):
        """returns a requests.Session to pass to hvac (or None)"""
        if not HAS_RETRIES:
            raise NotImplementedError('Retries are unavailable. This may indicate very old versions of one or more of the following: hvac, requests, urllib3.')

        class CallbackRetry(urllib3.util.Retry):

            def __init__(self, *args, **kwargs):
                self._newcb = kwargs.pop('new_callback')
                super(CallbackRetry, self).__init__(*args, **kwargs)

            def new(self, **kwargs):
                if self._newcb is not None:
                    self._newcb(self)
                kwargs['new_callback'] = self._newcb
                return super(CallbackRetry, self).new(**kwargs)
        if 'raise_on_status' not in retry_kwargs:
            retry_kwargs['raise_on_status'] = False
        retry = CallbackRetry(**retry_kwargs)
        adapter = HTTPAdapter(max_retries=retry)
        sess = Session()
        sess.mount('https://', adapter)
        sess.mount('http://', adapter)
        return sess

    def _process_option_retries(self):
        """check if retries option is int or dict and interpret it appropriately"""
        retries_opt = self._options.get_option('retries')
        if retries_opt is None:
            return
        retries = self._RETRIES_DEFAULT_PARAMS.copy()
        try:
            retries_int = check_type_int(retries_opt)
            if retries_int < 0:
                raise ValueError('Number of retries must be >= 0 (got %i)' % retries_int)
            elif retries_int == 0:
                retries = None
            else:
                retries['total'] = retries_int
        except TypeError:
            try:
                retries = check_type_dict(retries_opt)
            except TypeError:
                raise TypeError('retries option must be interpretable as int or dict. Got: %r' % retries_opt)
        self._options.set_option('retries', retries)

    def _process_option_proxies(self):
        """check if 'proxies' option is dict or str and set it appropriately"""
        proxies_opt = self._options.get_option('proxies')
        if proxies_opt is None:
            return
        try:
            proxies = check_type_dict(proxies_opt)
        except TypeError:
            proxy = check_type_str(proxies_opt)
            proxies = {'http': proxy, 'https': proxy}
        self._options.set_option('proxies', proxies)

    def _boolean_or_cacert(self):
        """return a bool or cacert"""
        ca_cert = self._options.get_option('ca_cert')
        validate_certs = self._options.get_option('validate_certs')
        if validate_certs is None:
            vault_skip_verify = os.environ.get('VAULT_SKIP_VERIFY')
            if vault_skip_verify is not None:
                try:
                    vault_skip_verify = check_type_bool(vault_skip_verify)
                except TypeError:
                    validate_certs = True
                else:
                    validate_certs = not vault_skip_verify
            else:
                validate_certs = True
        if not (validate_certs and ca_cert):
            self._conopt_verify = validate_certs
        else:
            self._conopt_verify = to_text(ca_cert, errors='surrogate_or_strict')
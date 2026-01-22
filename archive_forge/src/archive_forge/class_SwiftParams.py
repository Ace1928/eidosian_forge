import configparser
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.i18n import _, _LE
class SwiftParams(object):

    def __init__(self):
        if is_multiple_swift_store_accounts_enabled():
            self.params = self._load_config()
        else:
            self.params = self._form_default_params()

    def _form_default_params(self):
        default = {}
        if CONF.swift_store_user and CONF.swift_store_key and CONF.swift_store_auth_address:
            default['user'] = CONF.swift_store_user
            default['key'] = CONF.swift_store_key
            default['auth_address'] = CONF.swift_store_auth_address
            return {CONF.default_swift_reference: default}
        return {}

    def _load_config(self):
        try:
            conf_file = CONF.find_file(CONF.swift_store_config_file)
            CONFIG.read(conf_file)
        except Exception as e:
            msg = _LE('swift config file %(conf_file)s:%(exc)s not found') % {'conf_file': CONF.swift_store_config_file, 'exc': e}
            LOG.error(msg)
            raise exception.InvalidSwiftStoreConfiguration()
        account_params = {}
        account_references = CONFIG.sections()
        for ref in account_references:
            reference = {}
            try:
                reference['auth_address'] = CONFIG.get(ref, 'auth_address')
                reference['user'] = CONFIG.get(ref, 'user')
                reference['key'] = CONFIG.get(ref, 'key')
                account_params[ref] = reference
            except (ValueError, SyntaxError, configparser.NoOptionError):
                LOG.exception(_LE('Invalid format of swift store config cfg'))
        return account_params
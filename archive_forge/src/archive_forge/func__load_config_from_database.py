import copy
import functools
import itertools
import operator
import os
import threading
import uuid
from oslo_config import cfg
from oslo_log import log
from pycadf import reason
from keystone import assignment  # TODO(lbragstad): Decouple this dependency
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping
from keystone import notifications
from oslo_utils import timeutils
def _load_config_from_database(self, domain_id, specific_config):

    def _assert_no_more_than_one_sql_driver(domain_id, new_config):
        """Ensure adding driver doesn't push us over the limit of 1.

            The checks we make in this method need to take into account that
            we may be in a multiple process configuration and ensure that
            any race conditions are avoided.

            """
        if not new_config['driver'].is_sql:
            PROVIDERS.domain_config_api.release_registration(domain_id)
            return
        domain_registered = 'Unknown'
        for attempt in range(REGISTRATION_ATTEMPTS):
            if PROVIDERS.domain_config_api.obtain_registration(domain_id, SQL_DRIVER):
                LOG.debug('Domain %s successfully registered to use the SQL driver.', domain_id)
                return
            try:
                domain_registered = PROVIDERS.domain_config_api.read_registration(SQL_DRIVER)
            except exception.ConfigRegistrationNotFound:
                msg = 'While attempting to register domain %(domain)s to use the SQL driver, another process released it, retrying (attempt %(attempt)s).'
                LOG.debug(msg, {'domain': domain_id, 'attempt': attempt + 1})
                continue
            if domain_registered == domain_id:
                LOG.debug('While attempting to register domain %s to use the SQL driver, found that another process had already registered this domain. This is normal in multi-process configurations.', domain_id)
                return
            try:
                PROVIDERS.resource_api.get_domain(domain_registered)
            except exception.DomainNotFound:
                msg = 'While attempting to register domain %(domain)s to use the SQL driver, found that it was already registered to a domain that no longer exists (%(old_domain)s). Removing this stale registration and retrying (attempt %(attempt)s).'
                LOG.debug(msg, {'domain': domain_id, 'old_domain': domain_registered, 'attempt': attempt + 1})
                PROVIDERS.domain_config_api.release_registration(domain_registered, type=SQL_DRIVER)
                continue
            details = _('Config API entity at /domains/%s/config') % domain_id
            raise exception.MultipleSQLDriversInConfig(source=details)
        msg = _('Exceeded attempts to register domain %(domain)s to use the SQL driver, the last domain that appears to have had it is %(last_domain)s, giving up') % {'domain': domain_id, 'last_domain': domain_registered}
        raise exception.UnexpectedError(msg)
    domain_config = {}
    domain_config['cfg'] = cfg.ConfigOpts()
    keystone.conf.configure(conf=domain_config['cfg'])
    domain_config['cfg'](args=[], project='keystone', default_config_files=[], default_config_dirs=[])
    for group in specific_config:
        for option in specific_config[group]:
            domain_config['cfg'].set_override(option, specific_config[group][option], group)
    domain_config['cfg_overrides'] = specific_config
    domain_config['driver'] = self._load_driver(domain_config)
    _assert_no_more_than_one_sql_driver(domain_id, domain_config)
    self[domain_id] = domain_config
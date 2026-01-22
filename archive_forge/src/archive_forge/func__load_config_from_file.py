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
def _load_config_from_file(self, resource_api, file_list, domain_name):

    def _assert_no_more_than_one_sql_driver(new_config, config_file):
        """Ensure there is no more than one sql driver.

            Check to see if the addition of the driver in this new config
            would cause there to be more than one sql driver.

            """
        if new_config['driver'].is_sql and (self.driver.is_sql or self._any_sql):
            raise exception.MultipleSQLDriversInConfig(source=config_file)
        self._any_sql = self._any_sql or new_config['driver'].is_sql
    try:
        domain_ref = resource_api.get_domain_by_name(domain_name)
    except exception.DomainNotFound:
        LOG.warning('Invalid domain name (%s) found in config file name', domain_name)
        return
    domain_config = {}
    domain_config['cfg'] = cfg.ConfigOpts()
    keystone.conf.configure(conf=domain_config['cfg'])
    domain_config['cfg'](args=[], project='keystone', default_config_files=file_list, default_config_dirs=[])
    domain_config['driver'] = self._load_driver(domain_config)
    _assert_no_more_than_one_sql_driver(domain_config, file_list)
    self[domain_ref['id']] = domain_config
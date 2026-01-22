import copy
from unittest import mock
import uuid
import fixtures
import http.client
import ldap
from oslo_log import versionutils
import pkg_resources
from testtools import matchers
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import identity
from keystone.identity.backends import ldap as ldap_identity
from keystone.identity.backends.ldap import common as common_ldap
from keystone.identity.backends import sql as sql_identity
from keystone.identity.mapping_backends import mapping as map
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
from keystone.tests.unit.resource import test_backends as resource_tests
def _assert_backends(testcase, **kwargs):

    def _get_backend_cls(testcase, subsystem):
        observed_backend = getattr(testcase, subsystem + '_api').driver
        return observed_backend.__class__

    def _get_domain_specific_backend_cls(manager, domain):
        observed_backend = manager.domain_configs.get_domain_driver(domain)
        return observed_backend.__class__

    def _get_entrypoint_cls(subsystem, name):
        entrypoint = entrypoint_map['keystone.' + subsystem][name]
        return entrypoint.resolve()

    def _load_domain_specific_configs(manager):
        if not manager.domain_configs.configured and CONF.identity.domain_specific_drivers_enabled:
            manager.domain_configs.setup_domain_drivers(manager.driver, manager.resource_api)

    def _assert_equal(expected_cls, observed_cls, subsystem, domain=None):
        msg = 'subsystem %(subsystem)s expected %(expected_cls)r, but observed %(observed_cls)r'
        if domain:
            subsystem = '%s[domain=%s]' % (subsystem, domain)
        assert expected_cls == observed_cls, msg % {'expected_cls': expected_cls, 'observed_cls': observed_cls, 'subsystem': subsystem}
    env = pkg_resources.Environment()
    keystone_dist = env['keystone'][0]
    entrypoint_map = pkg_resources.get_entry_map(keystone_dist)
    for subsystem, entrypoint_name in kwargs.items():
        if isinstance(entrypoint_name, str):
            observed_cls = _get_backend_cls(testcase, subsystem)
            expected_cls = _get_entrypoint_cls(subsystem, entrypoint_name)
            _assert_equal(expected_cls, observed_cls, subsystem)
        elif isinstance(entrypoint_name, dict):
            manager = getattr(testcase, subsystem + '_api')
            _load_domain_specific_configs(manager)
            for domain, entrypoint_name in entrypoint_name.items():
                if domain is None:
                    observed_cls = _get_backend_cls(testcase, subsystem)
                    expected_cls = _get_entrypoint_cls(subsystem, entrypoint_name)
                    _assert_equal(expected_cls, observed_cls, subsystem)
                    continue
                observed_cls = _get_domain_specific_backend_cls(manager, domain)
                expected_cls = _get_entrypoint_cls(subsystem, entrypoint_name)
                _assert_equal(expected_cls, observed_cls, subsystem, domain)
        else:
            raise ValueError('%r is not an expected value for entrypoint name' % entrypoint_name)
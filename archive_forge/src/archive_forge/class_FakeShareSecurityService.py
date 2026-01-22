import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeShareSecurityService(object):
    """Fake one or more share security service"""

    @staticmethod
    def create_fake_security_service(attrs=None, methods=None):
        """Create a fake share security service

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
        attrs = attrs or {}
        methods = methods or {}
        share_security_service_info = {'created_at': datetime.datetime.now().isoformat(), 'description': 'description', 'dns_ip': '0.0.0.0', 'domain': 'fake.domain', 'id': uuid.uuid4().hex, 'name': 'name-' + uuid.uuid4().hex, 'ou': 'fake_OU', 'password': 'password', 'project_id': uuid.uuid4().hex, 'server': 'fake_hostname', 'default_ad_site': 'fake_default_ad_site', 'status': 'new', 'type': 'ldap', 'updated_at': datetime.datetime.now().isoformat(), 'user': 'fake_user'}
        share_security_service_info.update(attrs)
        share_security_service = osc_fakes.FakeResource(info=copy.deepcopy(share_security_service_info), methods=methods, loaded=True)
        return share_security_service

    @staticmethod
    def create_fake_security_services(attrs=None, count=2):
        """Create multiple fake security services.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share security services to be faked
        :return:
            A list of FakeResource objects
        """
        security_services = []
        for n in range(count):
            security_services.append(FakeShareSecurityService.create_fake_security_service(attrs))
        return security_services
import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeShareService(object):
    """Fake one or more share service"""

    @staticmethod
    def create_fake_service(attrs=None):
        """Create a fake share service

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
        attrs = attrs or {}
        share_service_info = {'binary': 'manila-share', 'host': 'fake_host@fake_backend', 'id': uuid.uuid4().hex, 'status': 'enabled', 'state': 'up', 'updated_at': 'time-' + uuid.uuid4().hex, 'zone': 'fake_zone'}
        share_service_info.update(attrs)
        share_service = osc_fakes.FakeResource(info=copy.deepcopy(share_service_info), loaded=True)
        return share_service

    @staticmethod
    def create_fake_services(attrs=None, count=2):
        """Create multiple fake services.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share services to be faked
        :return:
            A list of FakeResource objects
        """
        services = []
        for n in range(count):
            services.append(FakeShareService.create_fake_service(attrs))
        return services
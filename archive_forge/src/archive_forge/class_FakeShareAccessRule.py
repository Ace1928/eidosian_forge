import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeShareAccessRule(object):
    """Fake one or more share access rules"""

    @staticmethod
    def create_one_access_rule(attrs=None):
        """Create a fake share access rule

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
        share_access_rule = {'id': 'access_rule-id-' + uuid.uuid4().hex, 'share_id': 'share-id-' + uuid.uuid4().hex, 'access_level': 'rw', 'access_to': 'demo', 'access_type': 'user', 'state': 'active', 'access_key': None, 'created_at': datetime.datetime.now().isoformat(), 'updated_at': None, 'properties': {}}
        share_access_rule.update(attrs)
        share_access_rule = osc_fakes.FakeResource(info=copy.deepcopy(share_access_rule), loaded=True)
        return share_access_rule
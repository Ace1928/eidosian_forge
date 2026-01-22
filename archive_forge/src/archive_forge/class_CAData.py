from oslo_utils import timeutils
from barbicanclient.tests import test_client
from barbicanclient.v1 import cas
class CAData(object):

    def __init__(self, description='Test CA description'):
        self.name = 'Test CA'
        self.description = description
        self.plugin_name = 'Test CA Plugin'
        self.plugin_ca_id = 'plugin_uuid'
        now = timeutils.utcnow()
        self.expiration = str(now)
        self.created = str(now)
        self.meta = []
        self.meta.append({'name': self.name})
        if self.description:
            self.meta.append({'description': self.description})
        self.ca_dict = {'meta': self.meta, 'status': 'ACTIVE', 'plugin_name': self.plugin_name, 'plugin_ca_id': self.plugin_ca_id, 'created': self.created}

    def get_dict(self, ca_ref=None):
        ca = self.ca_dict
        if ca_ref:
            ca['ca_ref'] = ca_ref
        return ca
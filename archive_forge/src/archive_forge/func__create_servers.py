import datetime
from oslo_utils import timeutils
from novaclient.tests.functional import base
def _create_servers(self, name, number):
    return [self._create_server(name) for i in range(number)]
from octaviaclient.api import constants
from octaviaclient.api.v2 import octavia
from osc_lib import exceptions
from heat.engine.clients import client_plugin
from heat.engine import constraints
class OctaviaConstraint(constraints.BaseCustomConstraint):
    expected_exceptions = (exceptions.NotFound, octavia.OctaviaClientException)
    base_url = None

    def validate_with_client(self, client, value):
        octavia_client = client.client(CLIENT_NAME)
        octavia_client.find(path=self.base_url, value=value, attr=DEFAULT_FIND_ATTR)
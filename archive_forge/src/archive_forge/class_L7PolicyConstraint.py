from octaviaclient.api import constants
from octaviaclient.api.v2 import octavia
from osc_lib import exceptions
from heat.engine.clients import client_plugin
from heat.engine import constraints
class L7PolicyConstraint(OctaviaConstraint):
    base_url = constants.BASE_L7POLICY_URL
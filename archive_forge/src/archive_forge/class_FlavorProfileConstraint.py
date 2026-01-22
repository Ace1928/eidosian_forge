from octaviaclient.api import constants
from octaviaclient.api.v2 import octavia
from osc_lib import exceptions
from heat.engine.clients import client_plugin
from heat.engine import constraints
class FlavorProfileConstraint(OctaviaConstraint):
    base_url = constants.BASE_FLAVORPROFILE_URL
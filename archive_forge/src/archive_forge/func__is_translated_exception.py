from octaviaclient.api import constants
from octaviaclient.api.v2 import octavia
from osc_lib import exceptions
from heat.engine.clients import client_plugin
from heat.engine import constraints
def _is_translated_exception(ex, code):
    return isinstance(ex, octavia.OctaviaClientException) and ex.code == code
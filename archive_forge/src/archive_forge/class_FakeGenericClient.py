import datetime
from keystoneauth1 import session
from osc_lib.command import timing
from osc_lib.tests import fakes
from osc_lib.tests import utils
class FakeGenericClient(object):

    def __init__(self, **kwargs):
        self.auth_token = kwargs['token']
        self.management_url = kwargs['endpoint']
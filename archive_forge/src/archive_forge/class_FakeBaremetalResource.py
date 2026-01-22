import json
from unittest import mock
from osc_lib.tests import utils
from ironicclient.tests.unit.osc import fakes
class FakeBaremetalResource(fakes.FakeResource):

    def get_keys(self):
        return {'property': 'value'}
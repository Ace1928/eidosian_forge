import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def _build_hints(self, hints, filters, fed_dict):
    for key in filters:
        hints.add_filter(key, fed_dict[key], comparator='equals')
    return hints
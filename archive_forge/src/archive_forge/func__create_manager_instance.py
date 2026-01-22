import uuid
from keystone.common import manager
from keystone.common import provider_api
from keystone.tests import unit
def _create_manager_instance(self, provides_api=None):
    provides_api = provides_api or '%s_api' % uuid.uuid4().hex

    class TestManager(manager.Manager):
        _provides_api = provides_api
        driver_namespace = '_TEST_NOTHING'

        def do_something(self):
            return provides_api
    return TestManager(driver_name=None)
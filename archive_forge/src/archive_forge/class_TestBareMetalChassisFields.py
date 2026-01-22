from openstack import exceptions
from openstack.tests.functional.baremetal import base
class TestBareMetalChassisFields(base.BaseBaremetalTest):
    min_microversion = '1.8'

    def test_chassis_fields(self):
        self.create_chassis(description='something')
        result = self.conn.baremetal.chassis(fields=['uuid', 'extra'])
        for ch in result:
            self.assertIsNotNone(ch.id)
            self.assertIsNone(ch.description)
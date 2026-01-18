from unittest import mock
from designateclient import exceptions as designate_exception
from heat.common import exception
from heat.engine.resources.openstack.designate import recordset
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resource_validate_properties(self):
    mock_record_create = self.test_client_plugin.record_create
    mock_resource = self._get_mock_resource()
    mock_record_create.return_value = mock_resource
    self.assertEqual('test-record.com', self.test_resource.properties.get(recordset.DesignateRecordSet.NAME))
    self.assertEqual('Test record', self.test_resource.properties.get(recordset.DesignateRecordSet.DESCRIPTION))
    self.assertEqual(3600, self.test_resource.properties.get(recordset.DesignateRecordSet.TTL))
    self.assertEqual('A', self.test_resource.properties.get(recordset.DesignateRecordSet.TYPE))
    self.assertEqual(['1.1.1.1'], self.test_resource.properties.get(recordset.DesignateRecordSet.RECORDS))
    self.assertEqual('1234567', self.test_resource.properties.get(recordset.DesignateRecordSet.ZONE))
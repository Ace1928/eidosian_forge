from unittest import mock
from heat.common import exception
from heat.engine import environment
from heat.engine import resource as res
from heat.engine import service
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_resource_schema_unavailable(self):
    type_name = 'ResourceWithDefaultClientName'
    with mock.patch.object(generic_rsrc.ResourceWithDefaultClientName, 'is_service_available') as mock_is_service_available:
        mock_is_service_available.return_value = (False, 'Service endpoint not in service catalog.')
        ex = self.assertRaises(exception.ResourceTypeUnavailable, self.eng.resource_schema, self.ctx, type_name)
        msg = 'HEAT-E99001 Service sample is not available for resource type ResourceWithDefaultClientName, reason: Service endpoint not in service catalog.'
        self.assertEqual(msg, str(ex), 'invalid exception message')
        mock_is_service_available.assert_called_once_with(self.ctx)
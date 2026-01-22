from unittest import mock
import ddt
from cinderclient.tests.unit import utils
from cinderclient.v3 import limits
@ddt.ddt
class TestLimitsManager(utils.TestCase):

    @ddt.data(None, 'test')
    def test_get(self, tenant_id):
        api = mock.Mock()
        api.client.get.return_value = (None, {'limits': {'absolute': {'name1': 'value1'}}, 'no-limits': {'absolute': {'name2': 'value2'}}})
        l1 = limits.AbsoluteLimit('name1', 'value1')
        limitsManager = limits.LimitsManager(api)
        lim = limitsManager.get(tenant_id)
        query_str = ''
        if tenant_id:
            query_str = '?tenant_id=%s' % tenant_id
        api.client.get.assert_called_once_with('/limits%s' % query_str)
        self.assertIsInstance(lim, limits.Limits)
        for limit in lim.absolute:
            self.assertEqual(l1, limit)
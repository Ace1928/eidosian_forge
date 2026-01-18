from oslotest import base
from oslo_config import cfg
from oslo_config import fixture as config
def test_overridden_value(self):
    f = self._make_fixture()
    self.assertEqual(f.conf.get('testing_option'), 'initial_value')
    f.config(testing_option='changed_value')
    self.assertEqual('changed_value', f.conf.get('testing_option'))
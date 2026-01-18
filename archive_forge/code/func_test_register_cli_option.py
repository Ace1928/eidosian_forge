from oslotest import base
from oslo_config import cfg
from oslo_config import fixture as config
def test_register_cli_option(self):
    f = self._make_fixture()
    opt = cfg.StrOpt('new_test_opt', default='initial_value')
    f.register_cli_opt(opt)
    self.assertEqual(f.conf.get('new_test_opt'), opt.default)
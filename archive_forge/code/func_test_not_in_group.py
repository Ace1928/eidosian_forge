from oslo_config import cfg
from oslo_config import types
from oslo_messaging._drivers import common as drv_cmn
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
def test_not_in_group(self):
    group = 'oslo_messaging_rabbit'
    url = transport.TransportURL.parse(self.conf, 'rabbit:///?unknown_opt=4')
    self.assertRaises(cfg.NoSuchOptError, drv_cmn.ConfigOptsProxy, self.conf, url, group)
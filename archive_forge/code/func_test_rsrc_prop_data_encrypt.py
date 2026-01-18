from oslo_config import cfg
from heat.db import api as db_api
from heat.db import models
from heat.objects import resource_properties_data as rpd_object
from heat.tests import common
from heat.tests import utils
def test_rsrc_prop_data_encrypt(self):
    cfg.CONF.set_override('encrypt_parameters_and_properties', True)
    rpd_obj, db_obj = self._get_rpd_and_db_obj()
    self.assertNotEqual(db_obj['data'], self.data)
    for key in self.data:
        self.assertEqual('cryptography_decrypt_v1', db_obj['data'][key][0])
    self.assertEqual(self.data, rpd_obj['data'])
    rpd_obj = rpd_object.ResourcePropertiesData._from_db_object(rpd_object.ResourcePropertiesData(self.ctx), self.ctx, db_obj)
    self.assertEqual(self.data, rpd_obj['data'])
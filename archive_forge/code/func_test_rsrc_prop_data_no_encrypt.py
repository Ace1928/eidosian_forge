from oslo_config import cfg
from heat.db import api as db_api
from heat.db import models
from heat.objects import resource_properties_data as rpd_object
from heat.tests import common
from heat.tests import utils
def test_rsrc_prop_data_no_encrypt(self):
    cfg.CONF.set_override('encrypt_parameters_and_properties', False)
    rpd_obj, db_obj = self._get_rpd_and_db_obj()
    self.assertEqual(db_obj['data'], self.data)
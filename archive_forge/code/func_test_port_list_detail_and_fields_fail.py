import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_port_list_detail_and_fields_fail(self):
    self.assertRaises(exc.InvalidAttribute, self.mgr.list, detail=True, fields=['uuid', 'address'])
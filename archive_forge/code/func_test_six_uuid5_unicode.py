from unittest import mock
import uuid
from neutron_lib.placement import utils as place_utils
from neutron_lib.tests import _base as base
def test_six_uuid5_unicode(self):
    try:
        place_utils.six_uuid5(namespace=self._uuid_ns, name='unicode string')
    except Exception:
        self.fail('could not generate uuid')
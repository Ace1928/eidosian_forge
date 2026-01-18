from unittest import mock
from oslotest import base
from monascaclient.osc import migration as migr
from monascaclient.v2_0 import alarm_definitions as ad
from monascaclient.v2_0 import shell
@mock.patch('monascaclient.osc.migration.make_client')
def test_should_not_patch_unknown_severity(self, mc):
    ad_id = '0495340b-58fd-4e1c-932b-5e6f9cc96490'
    st = 'foo'
    raw_args = '{0} --severity {1}'.format(ad_id, st).split(' ')
    self._patch_test(mc, raw_args, called=False)
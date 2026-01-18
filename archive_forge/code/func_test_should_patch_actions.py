from unittest import mock
from oslotest import base
from monascaclient.osc import migration as migr
from monascaclient.v2_0 import alarm_definitions as ad
from monascaclient.v2_0 import shell
@mock.patch('monascaclient.osc.migration.make_client')
def test_should_patch_actions(self, mc):
    ad_id = '0495340b-58fd-4e1c-932b-5e6f9cc96490'
    ad_action_id = '16012650-0b62-4692-9103-2d04fe81cc93'
    actions = ['alarm-actions', 'ok-actions', 'undetermined-actions']
    for action in actions:
        raw_args = '{0} --{1} {2}'.format(ad_id, action, ad_action_id).split(' ')
        self._patch_test(mc, raw_args, **{'alarm_id': ad_id, action.replace('-', '_'): [ad_action_id]})
import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_networks
@ddt.data(*itertools.product(['2.25', '2.26'], ['fake share nw', _FakeShareNetwork()]))
@ddt.unpack
def test_update_share_network(self, microversion, share_nw):
    api_version = api_versions.APIVersion(microversion)
    values = self.values.copy()
    if api_version >= api_versions.APIVersion('2.26'):
        del values['nova_net_id']
    body_expected = {share_networks.RESOURCE_NAME: values}
    manager = share_networks.ShareNetworkManager(fakes.FakeClient(api_version=api_version))
    with mock.patch.object(manager, '_update', fakes.fake_update):
        result = manager.update(share_nw, **values)
        id = share_nw.id if hasattr(share_nw, 'id') else share_nw
        self.assertEqual(result['url'], share_networks.RESOURCE_PATH % id)
        self.assertEqual(result['resp_key'], share_networks.RESOURCE_NAME)
        self.assertEqual(result['body'], body_expected)
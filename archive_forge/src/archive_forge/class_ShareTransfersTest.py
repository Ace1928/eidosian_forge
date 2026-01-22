from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
class ShareTransfersTest(utils.TestCase):

    def test_create(self):
        cs.transfers.create('1234')
        cs.assert_called('POST', '/%s' % TRANSFER_URL, body={'transfer': {'share_id': '1234', 'name': None}})

    def test_get(self):
        transfer_id = '5678'
        cs.transfers.get(transfer_id)
        cs.assert_called('GET', '/%s/%s' % (TRANSFER_URL, transfer_id))

    def test_list(self):
        cs.transfers.list()
        cs.assert_called('GET', '/%s/detail' % TRANSFER_URL)

    def test_delete(self):
        cs.transfers.delete('5678')
        cs.assert_called('DELETE', '/%s/5678' % TRANSFER_URL)

    def test_accept(self):
        transfer_id = '5678'
        auth_key = '12345'
        cs.transfers.accept(transfer_id, auth_key)
        cs.assert_called('POST', '/%s/%s/accept' % (TRANSFER_URL, transfer_id))
import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
class ShareType(object):
    id = 'fake share type id'
    name = 'fake share type name'
import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def _share_instances(self):
    instances = {'share_instances': [fake_share_instance]}
    return (200, {}, instances)
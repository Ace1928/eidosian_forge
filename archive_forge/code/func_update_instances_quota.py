from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def update_instances_quota(self, **kw):
    return (200, {}, {'quotas': {'instances': 51}})
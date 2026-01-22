import copy
from openstack import exceptions
from openstack.tests.unit import base
class ZoneTestWrapper:

    def __init__(self, ut, attrs):
        self.remote_res = attrs
        self.ut = ut

    def get_create_response_json(self):
        return self.remote_res

    def get_get_response_json(self):
        return self.remote_res

    def __getitem__(self, key):
        """Dict access to be able to access properties easily"""
        return self.remote_res[key]

    def cmp(self, other):
        ut = self.ut
        me = self.remote_res
        for k, v in me.items():
            ut.assertEqual(v, other[k])
import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
class FakeNodeGroupManager(object):

    def list(self, cluster_id, limit=None, marker=None, sort_key=None, sort_dir=None, detail=False):
        pass

    def get(self, cluster_id, id):
        pass

    def create(self, cluster_id, **kwargs):
        pass

    def delete(self, cluster_id, id):
        pass

    def update(self, cluster_id, id, patch):
        pass
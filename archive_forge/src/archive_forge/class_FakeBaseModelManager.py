import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
class FakeBaseModelManager(object):

    def list(self, limit=None, marker=None, sort_key=None, sort_dir=None, detail=False):
        pass

    def get(self, id):
        pass

    def create(self, **kwargs):
        pass

    def delete(self, id):
        pass

    def update(self, id, patch):
        pass

    def rotate_ca(self, **kwargs):
        pass
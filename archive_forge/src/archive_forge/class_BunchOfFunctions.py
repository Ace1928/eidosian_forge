import taskflow.engines
from taskflow.patterns import linear_flow
from taskflow import task as base
from taskflow import test
class BunchOfFunctions(object):

    def __init__(self, values):
        self.values = values

    def run_one(self, *args, **kwargs):
        self.values.append('one')

    def revert_one(self, *args, **kwargs):
        self.values.append('revert one')

    def run_fail(self, *args, **kwargs):
        self.values.append('fail')
        raise RuntimeError('Woot!')
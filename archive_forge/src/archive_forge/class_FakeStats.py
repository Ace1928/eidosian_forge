from unittest import mock
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.stats import Stats
class FakeStats(Stats):

    def __init__(self, manager=None, info={}, **kwargs):
        Stats.__init__(self, manager=manager, info=info)
        self.clusters = kwargs.get('clusters', 0)
        self.nodes = kwargs.get('nodes', 0)
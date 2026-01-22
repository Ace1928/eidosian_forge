from unittest import mock
from heat.tests import common
from heat.tests import utils
def _stub_client(self):
    self.blazar_client_plugin.client = lambda: self.blazar_client
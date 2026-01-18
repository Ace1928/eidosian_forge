import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
def test_verify_requirements(self):
    self.em._load_one_plugin(self.mock_ep, False, (), {}, verify_requirements=True)
    self.mock_ep.load.assert_called_once_with()
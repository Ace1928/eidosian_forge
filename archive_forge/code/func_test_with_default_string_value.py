import argparse
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import loading
from keystoneauth1.loading import cli
from keystoneauth1.tests.unit.loading import utils
@utils.mock_plugin()
def test_with_default_string_value(self, m):
    name = uuid.uuid4().hex
    klass = loading.register_auth_argparse_arguments(self.p, [], default=name)
    self.assertIsInstance(klass, utils.MockLoader)
    m.assert_called_once_with(name)
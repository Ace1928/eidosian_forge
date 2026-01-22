import argparse
from unittest import mock
import testtools
from ironicclient.osc import plugin
from ironicclient.tests.unit.osc import fakes
from ironicclient.v1 import client
@mock.patch.object(plugin.utils, 'env', lambda x: None)
class ReplaceLatestVersionTest(testtools.TestCase):

    @mock.patch.object(plugin, 'OS_BAREMETAL_API_LATEST', new=False)
    def test___call___latest(self):
        parser = argparse.ArgumentParser()
        plugin.build_option_parser(parser)
        namespace = argparse.Namespace()
        parser.parse_known_args(['--os-baremetal-api-version', 'latest'], namespace)
        self.assertEqual(plugin.LATEST_VERSION, namespace.os_baremetal_api_version)
        self.assertTrue(plugin.OS_BAREMETAL_API_LATEST)

    @mock.patch.object(plugin, 'OS_BAREMETAL_API_LATEST', new=True)
    def test___call___specific_version(self):
        parser = argparse.ArgumentParser()
        plugin.build_option_parser(parser)
        namespace = argparse.Namespace()
        parser.parse_known_args(['--os-baremetal-api-version', '1.4'], namespace)
        self.assertEqual('1.4', namespace.os_baremetal_api_version)
        self.assertFalse(plugin.OS_BAREMETAL_API_LATEST)

    @mock.patch.object(plugin, 'OS_BAREMETAL_API_LATEST', new=True)
    def test___call___default(self):
        parser = argparse.ArgumentParser()
        plugin.build_option_parser(parser)
        namespace = argparse.Namespace()
        parser.parse_known_args([], namespace)
        self.assertEqual(plugin.LATEST_VERSION, namespace.os_baremetal_api_version)
        self.assertTrue(plugin.OS_BAREMETAL_API_LATEST)
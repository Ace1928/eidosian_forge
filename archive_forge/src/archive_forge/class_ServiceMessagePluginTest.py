import http.client as httplib
import io
from unittest import mock
import ddt
import requests
import suds
from oslo_vmware import exceptions
from oslo_vmware import service
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@ddt.ddt
class ServiceMessagePluginTest(base.TestCase):
    """Test class for ServiceMessagePlugin."""

    def setUp(self):
        super(ServiceMessagePluginTest, self).setUp()
        self.plugin = service.ServiceMessagePlugin()

    @ddt.data(('value', 'foo', 'string'), ('removeKey', '1', 'int'), ('removeKey', 'foo', 'string'))
    @ddt.unpack
    def test_add_attribute_for_value(self, name, text, expected_xsd_type):
        node = mock.Mock()
        node.name = name
        node.text = text
        self.plugin.add_attribute_for_value(node)
        node.set.assert_called_once_with('xsi:type', 'xsd:%s' % expected_xsd_type)

    def test_marshalled(self):
        context = mock.Mock()
        self.plugin.prune = mock.Mock()
        self.plugin.marshalled(context)
        self.plugin.prune.assert_called_once_with(context.envelope)
        context.envelope.walk.assert_called_once_with(self.plugin.add_attribute_for_value)
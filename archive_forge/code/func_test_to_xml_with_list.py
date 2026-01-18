import collections
import datetime
from lxml import etree
from oslo_serialization import jsonutils as json
import webob
from heat.common import serializers
from heat.tests import common
def test_to_xml_with_list(self):
    fixture = {'name': ['1', '2']}
    expected = b'<name><member>1</member><member>2</member></name>'
    actual = serializers.XMLResponseSerializer().to_xml(fixture)
    actual_xml_tree = etree.XML(actual)
    actual_xml_dict = self._recursive_dict(actual_xml_tree)
    expected_xml_tree = etree.XML(expected)
    expected_xml_dict = self._recursive_dict(expected_xml_tree)
    self.assertEqual(expected_xml_dict, actual_xml_dict)
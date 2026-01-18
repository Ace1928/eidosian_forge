import collections
import datetime
from lxml import etree
from oslo_serialization import jsonutils as json
import webob
from heat.common import serializers
from heat.tests import common
def test_to_xml_with_json_only_keys(self):
    fixture = collections.OrderedDict([('aresponse', collections.OrderedDict([('is_public', True), ('TemplateBody', {'name1': 'test'}), ('Metadata', {'name2': 'test2'})]))])
    expected = '<aresponse><is_public>True</is_public><TemplateBody>{"name1": "test"}</TemplateBody><Metadata>{"name2": "test2"}</Metadata></aresponse>'.encode('latin-1')
    actual = serializers.XMLResponseSerializer().to_xml(fixture)
    actual_xml_tree = etree.XML(actual)
    actual_xml_dict = self._recursive_dict(actual_xml_tree)
    expected_xml_tree = etree.XML(expected)
    expected_xml_dict = self._recursive_dict(expected_xml_tree)
    self.assertEqual(expected_xml_dict, actual_xml_dict)
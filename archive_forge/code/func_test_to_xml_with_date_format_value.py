import collections
import datetime
from lxml import etree
from oslo_serialization import jsonutils as json
import webob
from heat.common import serializers
from heat.tests import common
def test_to_xml_with_date_format_value(self):
    fixture = {'date': datetime.datetime(1, 3, 8, 2)}
    expected = b'<date>0001-03-08 02:00:00</date>'
    actual = serializers.XMLResponseSerializer().to_xml(fixture)
    self.assertEqual(expected, actual)
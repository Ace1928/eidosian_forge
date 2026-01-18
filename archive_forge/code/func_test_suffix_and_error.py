from tests.unit import unittest
import xml.dom.minidom
import xml.sax
from boto.s3.website import WebsiteConfiguration
from boto.s3.website import RedirectLocation
from boto.s3.website import RoutingRules
from boto.s3.website import Condition
from boto.s3.website import RoutingRules
from boto.s3.website import RoutingRule
from boto.s3.website import Redirect
from boto import handler
def test_suffix_and_error(self):
    config = WebsiteConfiguration(suffix='index.html', error_key='error.html')
    xml = config.to_xml()
    self.assertIn('<ErrorDocument><Key>error.html</Key></ErrorDocument>', xml)
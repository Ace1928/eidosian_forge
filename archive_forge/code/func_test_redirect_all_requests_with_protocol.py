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
def test_redirect_all_requests_with_protocol(self):
    location = RedirectLocation(hostname='example.com', protocol='https')
    config = WebsiteConfiguration(redirect_all_requests_to=location)
    xml = config.to_xml()
    self.assertIn('<RedirectAllRequestsTo><HostName>example.com</HostName><Protocol>https</Protocol></RedirectAllRequestsTo>', xml)
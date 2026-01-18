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
def test_routing_rules_to_host_on_404(self):
    x = pretty_print_xml
    rules = RoutingRules()
    condition = Condition(http_error_code=404)
    redirect = Redirect(hostname='example.com', replace_key_prefix='report-404/')
    rules.add_rule(RoutingRule(condition, redirect))
    config = WebsiteConfiguration(suffix='index.html', routing_rules=rules)
    xml = config.to_xml()
    expected_xml = '<?xml version="1.0" encoding="UTF-8"?>\n            <WebsiteConfiguration xmlns=\'http://s3.amazonaws.com/doc/2006-03-01/\'>\n              <IndexDocument>\n                <Suffix>index.html</Suffix>\n              </IndexDocument>\n              <RoutingRules>\n                <RoutingRule>\n                <Condition>\n                  <HttpErrorCodeReturnedEquals>404</HttpErrorCodeReturnedEquals>\n                </Condition>\n                <Redirect>\n                  <HostName>example.com</HostName>\n                  <ReplaceKeyPrefixWith>report-404/</ReplaceKeyPrefixWith>\n                </Redirect>\n                </RoutingRule>\n              </RoutingRules>\n            </WebsiteConfiguration>\n        '
    self.assertEqual(x(expected_xml), x(xml))
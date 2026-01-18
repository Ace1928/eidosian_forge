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
def test_routing_rules_key_prefix(self):
    x = pretty_print_xml
    rules = RoutingRules()
    condition = Condition(key_prefix='docs/')
    redirect = Redirect(replace_key_prefix='documents/')
    rules.add_rule(RoutingRule(condition, redirect))
    config = WebsiteConfiguration(suffix='index.html', routing_rules=rules)
    xml = config.to_xml()
    expected_xml = '<?xml version="1.0" encoding="UTF-8"?>\n            <WebsiteConfiguration xmlns=\'http://s3.amazonaws.com/doc/2006-03-01/\'>\n              <IndexDocument>\n                <Suffix>index.html</Suffix>\n              </IndexDocument>\n              <RoutingRules>\n                <RoutingRule>\n                <Condition>\n                  <KeyPrefixEquals>docs/</KeyPrefixEquals>\n                </Condition>\n                <Redirect>\n                  <ReplaceKeyPrefixWith>documents/</ReplaceKeyPrefixWith>\n                </Redirect>\n                </RoutingRule>\n              </RoutingRules>\n            </WebsiteConfiguration>\n        '
    self.assertEqual(x(expected_xml), x(xml))
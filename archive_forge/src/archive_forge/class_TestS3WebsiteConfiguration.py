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
class TestS3WebsiteConfiguration(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_suffix_only(self):
        config = WebsiteConfiguration(suffix='index.html')
        xml = config.to_xml()
        self.assertIn('<IndexDocument><Suffix>index.html</Suffix></IndexDocument>', xml)

    def test_suffix_and_error(self):
        config = WebsiteConfiguration(suffix='index.html', error_key='error.html')
        xml = config.to_xml()
        self.assertIn('<ErrorDocument><Key>error.html</Key></ErrorDocument>', xml)

    def test_redirect_all_request_to_with_just_host(self):
        location = RedirectLocation(hostname='example.com')
        config = WebsiteConfiguration(redirect_all_requests_to=location)
        xml = config.to_xml()
        self.assertIn('<RedirectAllRequestsTo><HostName>example.com</HostName></RedirectAllRequestsTo>', xml)

    def test_redirect_all_requests_with_protocol(self):
        location = RedirectLocation(hostname='example.com', protocol='https')
        config = WebsiteConfiguration(redirect_all_requests_to=location)
        xml = config.to_xml()
        self.assertIn('<RedirectAllRequestsTo><HostName>example.com</HostName><Protocol>https</Protocol></RedirectAllRequestsTo>', xml)

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

    def test_key_prefix(self):
        x = pretty_print_xml
        rules = RoutingRules()
        condition = Condition(key_prefix='images/')
        redirect = Redirect(replace_key='folderdeleted.html')
        rules.add_rule(RoutingRule(condition, redirect))
        config = WebsiteConfiguration(suffix='index.html', routing_rules=rules)
        xml = config.to_xml()
        expected_xml = '<?xml version="1.0" encoding="UTF-8"?>\n            <WebsiteConfiguration xmlns=\'http://s3.amazonaws.com/doc/2006-03-01/\'>\n              <IndexDocument>\n                <Suffix>index.html</Suffix>\n              </IndexDocument>\n              <RoutingRules>\n                <RoutingRule>\n                <Condition>\n                  <KeyPrefixEquals>images/</KeyPrefixEquals>\n                </Condition>\n                <Redirect>\n                  <ReplaceKeyWith>folderdeleted.html</ReplaceKeyWith>\n                </Redirect>\n                </RoutingRule>\n              </RoutingRules>\n            </WebsiteConfiguration>\n        '
        self.assertEqual(x(expected_xml), x(xml))

    def test_builders(self):
        x = pretty_print_xml
        rules = RoutingRules()
        condition = Condition(http_error_code=404)
        redirect = Redirect(hostname='example.com', replace_key_prefix='report-404/')
        rules.add_rule(RoutingRule(condition, redirect))
        xml = rules.to_xml()
        rules2 = RoutingRules().add_rule(RoutingRule.when(http_error_code=404).then_redirect(hostname='example.com', replace_key_prefix='report-404/'))
        xml2 = rules2.to_xml()
        self.assertEqual(x(xml), x(xml2))

    def test_parse_xml(self):
        x = pretty_print_xml
        xml_in = '<?xml version="1.0" encoding="UTF-8"?>\n            <WebsiteConfiguration xmlns=\'http://s3.amazonaws.com/doc/2006-03-01/\'>\n              <IndexDocument>\n                <Suffix>index.html</Suffix>\n              </IndexDocument>\n              <ErrorDocument>\n                <Key>error.html</Key>\n              </ErrorDocument>\n              <RoutingRules>\n                <RoutingRule>\n                <Condition>\n                  <KeyPrefixEquals>docs/</KeyPrefixEquals>\n                </Condition>\n                <Redirect>\n                  <Protocol>https</Protocol>\n                  <HostName>www.example.com</HostName>\n                  <ReplaceKeyWith>documents/</ReplaceKeyWith>\n                  <HttpRedirectCode>302</HttpRedirectCode>\n                </Redirect>\n                </RoutingRule>\n                <RoutingRule>\n                <Condition>\n                  <HttpErrorCodeReturnedEquals>404</HttpErrorCodeReturnedEquals>\n                </Condition>\n                <Redirect>\n                  <HostName>example.com</HostName>\n                  <ReplaceKeyPrefixWith>report-404/</ReplaceKeyPrefixWith>\n                </Redirect>\n                </RoutingRule>\n              </RoutingRules>\n            </WebsiteConfiguration>\n        '
        webconfig = WebsiteConfiguration()
        h = handler.XmlHandler(webconfig, None)
        xml.sax.parseString(xml_in.encode('utf-8'), h)
        xml_out = webconfig.to_xml()
        self.assertEqual(x(xml_in), x(xml_out))
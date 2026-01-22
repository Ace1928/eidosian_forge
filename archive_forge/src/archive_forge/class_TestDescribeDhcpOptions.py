from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, DhcpOptions
class TestDescribeDhcpOptions(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DescribeDhcpOptionsResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n              <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n              <dhcpOptionsSet>\n                <item>\n                  <dhcpOptionsId>dopt-7a8b9c2d</dhcpOptionsId>\n                  <dhcpConfigurationSet>\n                    <item>\n                      <key>domain-name</key>\n                      <valueSet>\n                        <item>\n                          <value>example.com</value>\n                        </item>\n                      </valueSet>\n                    </item>\n                    <item>\n                      <key>domain-name-servers</key>\n                      <valueSet>\n                        <item>\n                          <value>10.2.5.1</value>\n                      </item>\n                      </valueSet>\n                    </item>\n                    <item>\n                      <key>domain-name-servers</key>\n                      <valueSet>\n                        <item>\n                          <value>10.2.5.2</value>\n                          </item>\n                      </valueSet>\n                    </item>\n                  </dhcpConfigurationSet>\n                  <tagSet/>\n                </item>\n              </dhcpOptionsSet>\n            </DescribeDhcpOptionsResponse>\n        '

    def test_get_all_dhcp_options(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.get_all_dhcp_options(['dopt-7a8b9c2d'], [('key', 'domain-name')])
        self.assert_request_parameters({'Action': 'DescribeDhcpOptions', 'DhcpOptionsId.1': 'dopt-7a8b9c2d', 'Filter.1.Name': 'key', 'Filter.1.Value.1': 'domain-name'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(len(api_response), 1)
        self.assertIsInstance(api_response[0], DhcpOptions)
        self.assertEquals(api_response[0].id, 'dopt-7a8b9c2d')
        self.assertEquals(api_response[0].options['domain-name'], ['example.com'])
        self.assertEquals(api_response[0].options['domain-name-servers'], ['10.2.5.1', '10.2.5.2'])
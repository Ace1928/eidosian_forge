from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection
def test_replace_network_acl(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.replace_network_acl_entry('acl-2cb85d45', 110, 'tcp', 'deny', '0.0.0.0/0', egress=False, port_range_from=139, port_range_to=139)
    self.assert_request_parameters({'Action': 'ReplaceNetworkAclEntry', 'NetworkAclId': 'acl-2cb85d45', 'RuleNumber': 110, 'Protocol': 'tcp', 'RuleAction': 'deny', 'Egress': 'false', 'CidrBlock': '0.0.0.0/0', 'PortRange.From': 139, 'PortRange.To': 139}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(response, True)
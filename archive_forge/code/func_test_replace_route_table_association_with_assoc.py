from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
def test_replace_route_table_association_with_assoc(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.replace_route_table_association_with_assoc('rtbassoc-faad4893', 'rtb-f9ad4890')
    self.assert_request_parameters({'Action': 'ReplaceRouteTableAssociation', 'AssociationId': 'rtbassoc-faad4893', 'RouteTableId': 'rtb-f9ad4890'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(api_response, 'rtbassoc-faad4893')
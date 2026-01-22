from datetime import datetime, timedelta
from mock import MagicMock, Mock
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
import boto.ec2
from boto.regioninfo import RegionInfo
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from boto.ec2.connection import EC2Connection
from boto.ec2.snapshot import Snapshot
from boto.ec2.reservedinstance import ReservedInstancesConfiguration
from boto.compat import http_client
class TestPurchaseReservedInstanceOffering(TestEC2ConnectionBase):

    def default_body(self):
        return b'<PurchaseReservedInstancesOffering />'

    def test_serialized_api_args(self):
        self.set_http_response(status_code=200)
        response = self.ec2.purchase_reserved_instance_offering('offering_id', 1, (100.0, 'USD'))
        self.assert_request_parameters({'Action': 'PurchaseReservedInstancesOffering', 'InstanceCount': 1, 'ReservedInstancesOfferingId': 'offering_id', 'LimitPrice.Amount': '100.0', 'LimitPrice.CurrencyCode': 'USD'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
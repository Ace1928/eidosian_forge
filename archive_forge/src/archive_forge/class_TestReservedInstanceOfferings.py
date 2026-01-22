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
class TestReservedInstanceOfferings(TestEC2ConnectionBase):

    def default_body(self):
        return b'\n            <DescribeReservedInstancesOfferingsResponse>\n                <requestId>d3253568-edcf-4897-9a3d-fb28e0b3fa38</requestId>\n                    <reservedInstancesOfferingsSet>\n                    <item>\n                        <reservedInstancesOfferingId>2964d1bf71d8</reservedInstancesOfferingId>\n                        <instanceType>c1.medium</instanceType>\n                        <availabilityZone>us-east-1c</availabilityZone>\n                        <duration>94608000</duration>\n                        <fixedPrice>775.0</fixedPrice>\n                        <usagePrice>0.0</usagePrice>\n                        <productDescription>product description</productDescription>\n                        <instanceTenancy>default</instanceTenancy>\n                        <currencyCode>USD</currencyCode>\n                        <offeringType>Heavy Utilization</offeringType>\n                        <recurringCharges>\n                            <item>\n                                <frequency>Hourly</frequency>\n                                <amount>0.095</amount>\n                            </item>\n                        </recurringCharges>\n                        <marketplace>false</marketplace>\n                        <pricingDetailsSet>\n                            <item>\n                                <price>0.045</price>\n                                <count>1</count>\n                            </item>\n                        </pricingDetailsSet>\n                    </item>\n                    <item>\n                        <reservedInstancesOfferingId>2dce26e46889</reservedInstancesOfferingId>\n                        <instanceType>c1.medium</instanceType>\n                        <availabilityZone>us-east-1c</availabilityZone>\n                        <duration>94608000</duration>\n                        <fixedPrice>775.0</fixedPrice>\n                        <usagePrice>0.0</usagePrice>\n                        <productDescription>Linux/UNIX</productDescription>\n                        <instanceTenancy>default</instanceTenancy>\n                        <currencyCode>USD</currencyCode>\n                        <offeringType>Heavy Utilization</offeringType>\n                        <recurringCharges>\n                            <item>\n                                <frequency>Hourly</frequency>\n                                <amount>0.035</amount>\n                            </item>\n                        </recurringCharges>\n                        <marketplace>false</marketplace>\n                        <pricingDetailsSet/>\n                    </item>\n                </reservedInstancesOfferingsSet>\n                <nextToken>next_token</nextToken>\n            </DescribeReservedInstancesOfferingsResponse>\n        '

    def test_get_reserved_instance_offerings(self):
        self.set_http_response(status_code=200)
        response = self.ec2.get_all_reserved_instances_offerings()
        self.assertEqual(len(response), 2)
        instance = response[0]
        self.assertEqual(instance.id, '2964d1bf71d8')
        self.assertEqual(instance.instance_type, 'c1.medium')
        self.assertEqual(instance.availability_zone, 'us-east-1c')
        self.assertEqual(instance.duration, 94608000)
        self.assertEqual(instance.fixed_price, '775.0')
        self.assertEqual(instance.usage_price, '0.0')
        self.assertEqual(instance.description, 'product description')
        self.assertEqual(instance.instance_tenancy, 'default')
        self.assertEqual(instance.currency_code, 'USD')
        self.assertEqual(instance.offering_type, 'Heavy Utilization')
        self.assertEqual(len(instance.recurring_charges), 1)
        self.assertEqual(instance.recurring_charges[0].frequency, 'Hourly')
        self.assertEqual(instance.recurring_charges[0].amount, '0.095')
        self.assertEqual(len(instance.pricing_details), 1)
        self.assertEqual(instance.pricing_details[0].price, '0.045')
        self.assertEqual(instance.pricing_details[0].count, '1')

    def test_get_reserved_instance_offerings_params(self):
        self.set_http_response(status_code=200)
        self.ec2.get_all_reserved_instances_offerings(reserved_instances_offering_ids=['id1', 'id2'], instance_type='t1.micro', availability_zone='us-east-1', product_description='description', instance_tenancy='dedicated', offering_type='offering_type', include_marketplace=False, min_duration=100, max_duration=1000, max_instance_count=1, next_token='next_token', max_results=10)
        self.assert_request_parameters({'Action': 'DescribeReservedInstancesOfferings', 'ReservedInstancesOfferingId.1': 'id1', 'ReservedInstancesOfferingId.2': 'id2', 'InstanceType': 't1.micro', 'AvailabilityZone': 'us-east-1', 'ProductDescription': 'description', 'InstanceTenancy': 'dedicated', 'OfferingType': 'offering_type', 'IncludeMarketplace': 'false', 'MinDuration': '100', 'MaxDuration': '1000', 'MaxInstanceCount': '1', 'NextToken': 'next_token', 'MaxResults': '10'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
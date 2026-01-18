import base64
from tests.compat import unittest, mock
from tests.unit import AWSMockServiceTestCase
from boto.ec2.connection import EC2Connection
def test_run_instances_user_data(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.run_instances(image_id='123456', instance_type='m1.large', security_groups=['group1', 'group2'], user_data='#!/bin/bash')
    self.assert_request_parameters({'Action': 'RunInstances', 'ImageId': '123456', 'InstanceType': 'm1.large', 'UserData': base64.b64encode(b'#!/bin/bash').decode('utf-8'), 'MaxCount': 1, 'MinCount': 1, 'SecurityGroup.1': 'group1', 'SecurityGroup.2': 'group2'}, ignore_params_values=['Version', 'AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp'])
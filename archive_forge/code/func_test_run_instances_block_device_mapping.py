from tests.compat import unittest
from boto.ec2.connection import EC2Connection
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from tests.compat import OrderedDict
from tests.unit import AWSMockServiceTestCase
def test_run_instances_block_device_mapping(self):
    self.set_http_response(status_code=200)
    dev_sdf = BlockDeviceType(snapshot_id='snap-12345')
    dev_sdg = BlockDeviceType(snapshot_id='snap-12346', delete_on_termination=True, encrypted=True)

    class OrderedBlockDeviceMapping(OrderedDict, BlockDeviceMapping):
        pass
    bdm = OrderedBlockDeviceMapping()
    bdm.update(OrderedDict((('/dev/sdf', dev_sdf), ('/dev/sdg', dev_sdg))))
    response = self.service_connection.run_instances(image_id='123456', instance_type='m1.large', security_groups=['group1', 'group2'], block_device_map=bdm)
    self.assert_request_parameters({'Action': 'RunInstances', 'BlockDeviceMapping.1.DeviceName': '/dev/sdf', 'BlockDeviceMapping.1.Ebs.DeleteOnTermination': 'false', 'BlockDeviceMapping.1.Ebs.SnapshotId': 'snap-12345', 'BlockDeviceMapping.2.DeviceName': '/dev/sdg', 'BlockDeviceMapping.2.Ebs.DeleteOnTermination': 'true', 'BlockDeviceMapping.2.Ebs.SnapshotId': 'snap-12346', 'BlockDeviceMapping.2.Ebs.Encrypted': 'true', 'ImageId': '123456', 'InstanceType': 'm1.large', 'MaxCount': 1, 'MinCount': 1, 'SecurityGroup.1': 'group1', 'SecurityGroup.2': 'group2'}, ignore_params_values=['Version', 'AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp'])
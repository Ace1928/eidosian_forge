import time
import boto
from boto.compat import six
from tests.compat import unittest
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
class TestVPCConnection(unittest.TestCase):

    def setUp(self):
        self.instances = []
        self.post_terminate_cleanups = []
        self.api = boto.connect_vpc()
        self.vpc = self.api.create_vpc('10.0.0.0/16')
        time.sleep(5)
        self.subnet = self.api.create_subnet(self.vpc.id, '10.0.0.0/24')
        self.post_terminate_cleanups.append((self.api.delete_subnet, (self.subnet.id,)))
        time.sleep(10)

    def post_terminate_cleanup(self):
        """Helper to run clean up tasks after instances are removed."""
        for fn, args in self.post_terminate_cleanups:
            fn(*args)
            time.sleep(10)
        if self.vpc:
            self.api.delete_vpc(self.vpc.id)

    def terminate_instances(self):
        """Helper to remove all instances and kick off additional cleanup
        once they are terminated.
        """
        for instance in self.instances:
            self.terminate_instance(instance)
        self.post_terminate_cleanup()

    def terminate_instance(self, instance):
        instance.terminate()
        for i in six.moves.range(300):
            instance.update()
            if instance.state == 'terminated':
                time.sleep(30)
                return
            else:
                time.sleep(10)

    def delete_elastic_ip(self, eip):
        new_eip = self.api.get_all_addresses([eip.public_ip])[0]
        if new_eip.association_id:
            new_eip.disassociate()
        new_eip.release()
        time.sleep(10)

    def test_multi_ip_create(self):
        interface = NetworkInterfaceSpecification(device_index=0, subnet_id=self.subnet.id, private_ip_address='10.0.0.21', description='This is a test interface using boto.', delete_on_termination=True, private_ip_addresses=[PrivateIPAddress(private_ip_address='10.0.0.22', primary=False), PrivateIPAddress(private_ip_address='10.0.0.23', primary=False), PrivateIPAddress(private_ip_address='10.0.0.24', primary=False)])
        interfaces = NetworkInterfaceCollection(interface)
        reservation = self.api.run_instances(image_id='ami-a0cd60c9', instance_type='m1.small', network_interfaces=interfaces)
        time.sleep(10)
        instance = reservation.instances[0]
        self.addCleanup(self.terminate_instance, instance)
        retrieved = self.api.get_all_reservations(instance_ids=[instance.id])
        self.assertEqual(len(retrieved), 1)
        retrieved_instances = retrieved[0].instances
        self.assertEqual(len(retrieved_instances), 1)
        retrieved_instance = retrieved_instances[0]
        self.assertEqual(len(retrieved_instance.interfaces), 1)
        interface = retrieved_instance.interfaces[0]
        private_ip_addresses = interface.private_ip_addresses
        self.assertEqual(len(private_ip_addresses), 4)
        self.assertEqual(private_ip_addresses[0].private_ip_address, '10.0.0.21')
        self.assertEqual(private_ip_addresses[0].primary, True)
        self.assertEqual(private_ip_addresses[1].private_ip_address, '10.0.0.22')
        self.assertEqual(private_ip_addresses[2].private_ip_address, '10.0.0.23')
        self.assertEqual(private_ip_addresses[3].private_ip_address, '10.0.0.24')

    def test_associate_public_ip(self):
        interface = NetworkInterfaceSpecification(associate_public_ip_address=True, subnet_id=self.subnet.id, delete_on_termination=True)
        interfaces = NetworkInterfaceCollection(interface)
        reservation = self.api.run_instances(image_id='ami-a0cd60c9', instance_type='m1.small', network_interfaces=interfaces)
        instance = reservation.instances[0]
        self.instances.append(instance)
        self.addCleanup(self.terminate_instances)
        time.sleep(60)
        retrieved = self.api.get_all_reservations(instance_ids=[instance.id])
        self.assertEqual(len(retrieved), 1)
        retrieved_instances = retrieved[0].instances
        self.assertEqual(len(retrieved_instances), 1)
        retrieved_instance = retrieved_instances[0]
        self.assertEqual(len(retrieved_instance.interfaces), 1)
        interface = retrieved_instance.interfaces[0]
        self.assertTrue(interface.publicIp.count('.') >= 3)

    def test_associate_elastic_ip(self):
        interface = NetworkInterfaceSpecification(associate_public_ip_address=False, subnet_id=self.subnet.id, delete_on_termination=True)
        interfaces = NetworkInterfaceCollection(interface)
        reservation = self.api.run_instances(image_id='ami-a0cd60c9', instance_type='m1.small', network_interfaces=interfaces)
        instance = reservation.instances[0]
        self.instances.append(instance)
        self.addCleanup(self.terminate_instances)
        igw = self.api.create_internet_gateway()
        time.sleep(5)
        self.api.attach_internet_gateway(igw.id, self.vpc.id)
        self.post_terminate_cleanups.append((self.api.detach_internet_gateway, (igw.id, self.vpc.id)))
        self.post_terminate_cleanups.append((self.api.delete_internet_gateway, (igw.id,)))
        eip = self.api.allocate_address('vpc')
        self.post_terminate_cleanups.append((self.delete_elastic_ip, (eip,)))
        time.sleep(60)
        eip.associate(instance.id)
import time
from testtools import content
from openstack.tests.functional import base
def test_update_cluster_receiver(self):
    profile_name = 'test_profile'
    spec = {'properties': {'flavor': self.flavor.name, 'image': self.image.name, 'networks': [{'network': 'private'}], 'security_groups': ['default']}, 'type': 'os.nova.server', 'version': 1.0}
    self.addDetail('profile', content.text_content(profile_name))
    profile = self.user_cloud.create_cluster_profile(name=profile_name, spec=spec)
    self.addCleanup(self.cleanup_profile, profile['id'])
    cluster_name = 'example_cluster'
    desired_capacity = 0
    self.addDetail('cluster', content.text_content(cluster_name))
    cluster = self.user_cloud.create_cluster(name=cluster_name, profile=profile, desired_capacity=desired_capacity)
    self.addCleanup(self.cleanup_cluster, cluster['cluster']['id'])
    receiver_name = 'example_receiver'
    receiver_type = 'webhook'
    self.addDetail('receiver', content.text_content(receiver_name))
    receiver = self.user_cloud.create_cluster_receiver(name=receiver_name, receiver_type=receiver_type, cluster_name_or_id=cluster['cluster']['id'], action='CLUSTER_SCALE_OUT')
    self.addCleanup(self.cleanup_receiver, receiver['id'])
    receiver_update = self.user_cloud.update_cluster_receiver(receiver['id'], new_name='new_receiver_name')
    self.assertEqual(receiver_update['receiver']['id'], receiver['id'])
    self.assertEqual(receiver_update['receiver']['type'], receiver['type'])
    self.assertEqual(receiver_update['receiver']['cluster_id'], receiver['cluster_id'])
    self.assertEqual(receiver_update['receiver']['name'], 'new_receiver_name')
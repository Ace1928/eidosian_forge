from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_qos_policy(self):
    try:
        self.operator_cloud.delete_qos_policy(self.policy['id'])
    except Exception as e:
        raise exceptions.SDKException(e)
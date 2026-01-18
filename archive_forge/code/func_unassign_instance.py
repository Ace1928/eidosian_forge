import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def unassign_instance(self, instance_id):
    """
        Unassigns a registered instance from all of it's layers. The
        instance remains in the stack as an unassigned instance and
        can be assigned to another layer, as needed. You cannot use
        this action with instances that were created with AWS
        OpsWorks.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type instance_id: string
        :param instance_id: The instance ID.

        """
    params = {'InstanceId': instance_id}
    return self.make_request(action='UnassignInstance', body=json.dumps(params))
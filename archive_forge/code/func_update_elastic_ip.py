import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def update_elastic_ip(self, elastic_ip, name=None):
    """
        Updates a registered Elastic IP address's name. For more
        information, see `Resource Management`_.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type elastic_ip: string
        :param elastic_ip: The address.

        :type name: string
        :param name: The new name.

        """
    params = {'ElasticIp': elastic_ip}
    if name is not None:
        params['Name'] = name
    return self.make_request(action='UpdateElasticIp', body=json.dumps(params))
import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def update_my_user_profile(self, ssh_public_key=None):
    """
        Updates a user's SSH public key.

        **Required Permissions**: To use this action, an IAM user must
        have self-management enabled or an attached policy that
        explicitly grants permissions. For more information on user
        permissions, see `Managing User Permissions`_.

        :type ssh_public_key: string
        :param ssh_public_key: The user's SSH public key.

        """
    params = {}
    if ssh_public_key is not None:
        params['SshPublicKey'] = ssh_public_key
    return self.make_request(action='UpdateMyUserProfile', body=json.dumps(params))
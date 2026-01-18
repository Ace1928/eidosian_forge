import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def update_rds_db_instance(self, rds_db_instance_arn, db_user=None, db_password=None):
    """
        Updates an Amazon RDS instance.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type rds_db_instance_arn: string
        :param rds_db_instance_arn: The Amazon RDS instance's ARN.

        :type db_user: string
        :param db_user: The master user name.

        :type db_password: string
        :param db_password: The database password.

        """
    params = {'RdsDbInstanceArn': rds_db_instance_arn}
    if db_user is not None:
        params['DbUser'] = db_user
    if db_password is not None:
        params['DbPassword'] = db_password
    return self.make_request(action='UpdateRdsDbInstance', body=json.dumps(params))
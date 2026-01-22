import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def add_source_identifier_to_subscription(self, subscription_name, source_identifier):
    """
        Adds a source identifier to an existing RDS event notification
        subscription.

        :type subscription_name: string
        :param subscription_name: The name of the RDS event notification
            subscription you want to add a source identifier to.

        :type source_identifier: string
        :param source_identifier:
        The identifier of the event source to be added. An identifier must
            begin with a letter and must contain only ASCII letters, digits,
            and hyphens; it cannot end with a hyphen or contain two consecutive
            hyphens.

        Constraints:


        + If the source type is a DB instance, then a `DBInstanceIdentifier`
              must be supplied.
        + If the source type is a DB security group, a `DBSecurityGroupName`
              must be supplied.
        + If the source type is a DB parameter group, a `DBParameterGroupName`
              must be supplied.
        + If the source type is a DB snapshot, a `DBSnapshotIdentifier` must be
              supplied.

        """
    params = {'SubscriptionName': subscription_name, 'SourceIdentifier': source_identifier}
    return self._make_request(action='AddSourceIdentifierToSubscription', verb='POST', path='/', params=params)
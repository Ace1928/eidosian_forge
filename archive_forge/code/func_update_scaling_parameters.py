import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudsearch2 import exceptions
def update_scaling_parameters(self, domain_name, scaling_parameters):
    """
        Configures scaling parameters for a domain. A domain's scaling
        parameters specify the desired search instance type and
        replication count. Amazon CloudSearch will still automatically
        scale your domain based on the volume of data and traffic, but
        not below the desired instance type and replication count. If
        the Multi-AZ option is enabled, these values control the
        resources used per Availability Zone. For more information,
        see `Configuring Scaling Options`_ in the Amazon CloudSearch
        Developer Guide .

        :type domain_name: string
        :param domain_name: A string that represents the name of a domain.
            Domain names are unique across the domains owned by an account
            within an AWS region. Domain names start with a letter or number
            and can contain the following characters: a-z (lowercase), 0-9, and
            - (hyphen).

        :type scaling_parameters: dict
        :param scaling_parameters: The desired instance type and desired number
            of replicas of each index partition.

        """
    params = {'DomainName': domain_name}
    self.build_complex_param(params, 'ScalingParameters', scaling_parameters)
    return self._make_request(action='UpdateScalingParameters', verb='POST', path='/', params=params)
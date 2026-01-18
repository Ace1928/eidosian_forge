from boto.ec2.elb.healthcheck import HealthCheck
from boto.ec2.elb.listener import Listener
from boto.ec2.elb.listelement import ListElement
from boto.ec2.elb.policies import Policies, OtherPolicy
from boto.ec2.elb.securitygroup import SecurityGroup
from boto.ec2.instanceinfo import InstanceInfo
from boto.resultset import ResultSet
from boto.compat import six

        Associates one or more security groups with the load balancer.
        The provided security groups will override any currently applied
        security groups.

        :type security_groups: string or List of strings
        :param security_groups: The name of the security group(s) to add.

        
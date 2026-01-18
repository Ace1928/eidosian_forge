from boto.exception import BotoClientError
from boto.ec2.ec2object import TaggedEC2Object
from boto.resultset import ResultSet
from boto.ec2.group import Group

        Detach this ENI from an EC2 instance.

        :type force: bool
        :param force: Forces detachment if the previous detachment
                      attempt did not occur cleanly.

        :rtype: bool
        :return: True if successful
        
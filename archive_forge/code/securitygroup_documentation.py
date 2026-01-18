from boto.ec2.ec2object import TaggedEC2Object
from boto.exception import BotoClientError

        Find all of the current instances that are running within this
        security group.

        :rtype: list of :class:`boto.ec2.instance.Instance`
        :return: A list of Instance objects
        
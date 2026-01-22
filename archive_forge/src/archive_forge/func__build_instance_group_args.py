import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def _build_instance_group_args(self, instance_group):
    """
        Takes an InstanceGroup; returns a dict that, when its keys are
        properly prefixed, can be used for describing InstanceGroups in
        RunJobFlow or AddInstanceGroups requests.
        """
    params = {'InstanceCount': instance_group.num_instances, 'InstanceRole': instance_group.role, 'InstanceType': instance_group.type, 'Name': instance_group.name, 'Market': instance_group.market}
    if instance_group.market == 'SPOT':
        params['BidPrice'] = instance_group.bidprice
    return params
import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def _build_instance_group_list_args(self, instance_groups):
    """
        Takes a list of InstanceGroups, or a single InstanceGroup. Returns
        a comparable dict for use in making a RunJobFlow or AddInstanceGroups
        request.
        """
    if not isinstance(instance_groups, list):
        instance_groups = [instance_groups]
    params = {}
    for i, instance_group in enumerate(instance_groups):
        ig_dict = self._build_instance_group_args(instance_group)
        for key, value in six.iteritems(ig_dict):
            params['InstanceGroups.member.%d.%s' % (i + 1, key)] = value
    return params
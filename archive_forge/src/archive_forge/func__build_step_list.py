import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def _build_step_list(self, steps):
    if not isinstance(steps, list):
        steps = [steps]
    params = {}
    for i, step in enumerate(steps):
        for key, value in six.iteritems(step):
            params['Steps.member.%s.%s' % (i + 1, key)] = value
    return params
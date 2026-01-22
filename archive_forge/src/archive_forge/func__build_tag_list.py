import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def _build_tag_list(self, tags):
    assert isinstance(tags, dict)
    params = {}
    for i, key_value in enumerate(sorted(six.iteritems(tags)), start=1):
        key, value = key_value
        current_prefix = 'Tags.member.%s' % i
        params['%s.Key' % current_prefix] = key
        if value:
            params['%s.Value' % current_prefix] = value
    return params
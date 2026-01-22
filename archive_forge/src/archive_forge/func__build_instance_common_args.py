import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def _build_instance_common_args(self, ec2_keyname, availability_zone, keep_alive, hadoop_version):
    """
        Takes a number of parameters used when starting a jobflow (as
        specified in run_jobflow() above). Returns a comparable dict for
        use in making a RunJobFlow request.
        """
    params = {'Instances.KeepJobFlowAliveWhenNoSteps': str(keep_alive).lower()}
    if hadoop_version:
        params['Instances.HadoopVersion'] = hadoop_version
    if ec2_keyname:
        params['Instances.Ec2KeyName'] = ec2_keyname
    if availability_zone:
        params['Instances.Placement.AvailabilityZone'] = availability_zone
    return params
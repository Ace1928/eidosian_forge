import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def add_jobflow_steps(self, jobflow_id, steps):
    """
        Adds steps to a jobflow

        :type jobflow_id: str
        :param jobflow_id: The job flow id
        :type steps: list(boto.emr.Step)
        :param steps: A list of steps to add to the job
        """
    if not isinstance(steps, list):
        steps = [steps]
    params = {}
    params['JobFlowId'] = jobflow_id
    step_args = [self._build_step_args(step) for step in steps]
    params.update(self._build_step_list(step_args))
    return self.get_object('AddJobFlowSteps', params, JobFlowStepList, verb='POST')
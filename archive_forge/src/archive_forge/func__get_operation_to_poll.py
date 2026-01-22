from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.transfer import jobs_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import scaled_integer
def _get_operation_to_poll(job_name, operation_name):
    """Returns operation name or last operation of job name."""
    if not job_name and (not operation_name) or (job_name and operation_name):
        raise ValueError('job_name or operation_name must be provided but not both.')
    if job_name:
        latest_operation_name = jobs_util.block_until_operation_created(job_name)
        log.status.Print('Latest Operation: {}'.format(latest_operation_name))
        return latest_operation_name
    return operation_name
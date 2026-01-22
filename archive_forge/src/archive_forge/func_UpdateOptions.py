from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import shutil
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.command_lib.builds import submit_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
@staticmethod
def UpdateOptions(job_id, project_id=None, region_id=None, min_num_workers=None, max_num_workers=None, worker_utilization_hint=None, unset_worker_utilization_hint=None):
    """Update pipeline options on a running job.

    You should specify at-least one (or both) of min_num_workers and
    max_num_workers.

    Args:
      job_id: ID of job to update
      project_id: Project of the job
      region_id: Region the job is in
      min_num_workers: Lower-bound for worker autoscaling
      max_num_workers: Upper-bound for worker autoscaling
      worker_utilization_hint: Target CPU utilization for worker autoscaling
      unset_worker_utilization_hint: Unsets worker_utilization_hint value

    Returns:
      The updated Job
    """
    project_id = project_id or GetProject()
    region_id = region_id or DATAFLOW_API_DEFAULT_REGION
    job = GetMessagesModule().Job(runtimeUpdatableParams=GetMessagesModule().RuntimeUpdatableParams(minNumWorkers=min_num_workers, maxNumWorkers=max_num_workers, workerUtilizationHint=None if unset_worker_utilization_hint else worker_utilization_hint))
    update_mask_pieces = []
    if min_num_workers is not None:
        update_mask_pieces.append('runtime_updatable_params.min_num_workers')
    if max_num_workers is not None:
        update_mask_pieces.append('runtime_updatable_params.max_num_workers')
    if worker_utilization_hint is not None or unset_worker_utilization_hint:
        update_mask_pieces.append('runtime_updatable_params.worker_utilization_hint')
    update_mask = ','.join(update_mask_pieces)
    request = GetMessagesModule().DataflowProjectsLocationsJobsUpdateRequest(jobId=job_id, location=region_id, projectId=project_id, job=job, updateMask=update_mask)
    try:
        return Jobs.GetService().Update(request)
    except apitools_exceptions.HttpError as error:
        raise exceptions.HttpException(error)
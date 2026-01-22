from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import config
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags
from googlecloudsdk.command_lib.builds import staging_bucket_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def DetermineBuildRegion(build_config, desired_region=None):
    """Determine what region of the GCB service this build should be sent to.

  Args:
    build_config: apitools.base.protorpclite.messages.Message, The Build message
      to analyze.
    desired_region: str, The region requested by the user, if any.

  Raises:
    RegionMismatchError: If the config conflicts with the desired region.

  Returns:
    str, The region that the build should be sent to, or None if it should be
    sent to the global region.

  Note: we do not validate region strings so that old versions of gcloud are
  able to access new regions. This is aligned with the approach used by other
  teams.
  """
    wp_options = build_config.options
    if not wp_options:
        return desired_region
    wp_resource = wp_options.pool.name if wp_options.pool else ''
    if not wp_resource:
        wp_resource = wp_options.workerPool
    if not wp_resource:
        return desired_region
    if not cloudbuild_util.IsWorkerPool(wp_resource):
        return desired_region
    wp_region = cloudbuild_util.WorkerPoolRegion(wp_resource)
    matches = []
    if build_config.substitutions and wp_region:
        substitution_keys = list((p.key for p in build_config.substitutions.additionalProperties))
        matches = [k in wp_region for k in substitution_keys]
    if not desired_region and wp_region and matches:
        raise c_exceptions.InvalidArgumentException('--region', '--region flag required when workerpool resource includes region substitution')
    if desired_region and desired_region != wp_region:
        return desired_region
    return wp_region
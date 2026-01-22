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
def CreateBuildConfigAlpha(tag, no_cache, messages, substitutions, arg_config, is_specified_source, no_source, source, gcs_source_staging_dir, ignore_file, arg_gcs_log_dir, arg_machine_type, arg_disk_size, arg_memory, arg_vcpu_count, arg_worker_pool, arg_dir, arg_revision, arg_git_source_dir, arg_git_source_revision, buildpack, hide_logs=False, arg_bucket_behavior=None, skip_set_source=False, client_tag=None):
    """Returns a build config."""
    timeout_str = _GetBuildTimeout()
    build_config = _SetBuildSteps(tag, no_cache, messages, substitutions, arg_config, no_source, source, timeout_str, buildpack, client_tag)
    if not skip_set_source:
        build_config = SetSource(build_config, messages, is_specified_source, no_source, source, gcs_source_staging_dir, arg_dir, arg_revision, arg_git_source_dir, arg_git_source_revision, ignore_file, hide_logs=hide_logs)
    build_config = _SetLogsBucket(build_config, arg_gcs_log_dir)
    build_config = _SetMachineType(build_config, messages, arg_machine_type)
    build_config = _SetWorkerPool(build_config, messages, arg_worker_pool)
    build_config = _SetWorkerPoolConfig(build_config, messages, arg_disk_size, arg_memory, arg_vcpu_count)
    build_config = _SetDefaultLogsBucketBehavior(build_config, messages, arg_bucket_behavior)
    if cloudbuild_util.WorkerPoolConfigIsSpecified(build_config) and (not cloudbuild_util.WorkerPoolIsSpecified(build_config)):
        raise cloudbuild_exceptions.WorkerConfigButNoWorkerpoolError
    if not cloudbuild_util.WorkerPoolIsSpecified(build_config):
        build_config = _SetDiskSize(build_config, messages, arg_disk_size)
    return build_config